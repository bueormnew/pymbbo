"""Módulo core CausalMatrixMerge — bloque de memoria matricial causal.

Este módulo implementa el bloque central de la arquitectura Causal Matrix Merge.
NO es un transformer. Utiliza una memoria matricial de tamaño fijo que comprime
contexto infinito mediante una regla de actualización afín. No hay atención sobre
el historial de tokens — cada token actualiza una matriz fija y lee de ella.

La memoria [batch, slots, state_dim] se actualiza por cada token mediante:
    M_t = decay * M_{t-1} + (1 - decay) * write

Esto habilita contexto infinito porque la memoria es de tamaño fijo
independientemente de la longitud de la secuencia.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CausalMatrixMergeConfig
from .state import MergeState


def rms_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalización RMS ligera sin escala aprendida.

    Computa x * rsqrt(mean(x^2) + eps) a lo largo de la última dimensión.

    Args:
        x: Tensor de forma arbitraria.
        eps: Epsilon para estabilidad numérica.

    Returns:
        Tensor normalizado con la misma forma que x.
    """
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True).clamp_min(eps))


class CausalMatrixMerge(nn.Module):
    """Bloque de merge causal matricial con memoria de tamaño fijo.

    Implementa actualización afín con decay gates, routing por slots,
    escritura de bajo rango, banco de checkpoints y lectura por atención.
    La memoria es de tamaño fijo sin importar la longitud de la secuencia,
    habilitando contexto infinito con complejidad O(1) por token.

    Args:
        config: CausalMatrixMergeConfig con todos los hiperparámetros.
    """

    def __init__(self, config: CausalMatrixMergeConfig) -> None:
        super().__init__()
        self.config = config
        self.model_dim = config.model_dim
        self.state_dim = config.state_dim
        self.num_slots = config.num_slots
        self.write_rank = config.write_rank
        self.num_checkpoints = config.num_checkpoints
        self.checkpoint_stride = config.checkpoint_stride
        self.use_residual_gate = config.use_residual_gate

        # Proyección de entrada
        self.in_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)

        # Gates y routing
        self.decay_proj = nn.Linear(config.model_dim, config.num_slots, bias=True)
        self.route_proj = nn.Linear(config.model_dim, config.num_slots, bias=True)

        # Escritura de bajo rango
        self.write_proj = nn.Linear(
            config.model_dim, config.write_rank * config.state_dim, bias=True
        )

        # Proyecciones de lectura
        self.query_proj = nn.Linear(config.model_dim, config.state_dim, bias=False)
        self.key_proj = nn.Linear(config.state_dim, config.state_dim, bias=False)
        self.value_proj = nn.Linear(config.state_dim, config.model_dim, bias=False)

        # Salida
        self.out_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)

        # Puerta residual (opcional)
        if config.use_residual_gate:
            self.residual_gate = nn.Linear(config.model_dim, config.model_dim, bias=True)
        else:
            self.residual_gate = None

        # Normalización post-update
        self.post_norm = nn.LayerNorm(config.state_dim)

        # Dropout
        self.dropout_layer = nn.Dropout(config.dropout)

        # Contador de pasos (buffer no-entrenable)
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> MergeState:
        """Crea un estado inicial batched para procesamiento.

        Args:
            batch_size: Tamaño del batch.
            device: Dispositivo para los tensores. Si None, usa el dispositivo
                del módulo.
            dtype: Tipo de dato para los tensores. Si None, usa float32.

        Returns:
            MergeState con tensores batched:
                memory [B, S, Ds], normalizer [B, S, 1],
                checkpoints [B, K, S, Ds], step=0.
        """
        if device is None:
            device = self._step.device
        if dtype is None:
            dtype = torch.float32

        memory = torch.zeros(
            batch_size, self.num_slots, self.state_dim,
            device=device, dtype=dtype,
        )
        normalizer = torch.ones(
            batch_size, self.num_slots, 1,
            device=device, dtype=dtype,
        )
        checkpoints = torch.zeros(
            batch_size, self.num_checkpoints, self.num_slots, self.state_dim,
            device=device, dtype=dtype,
        )
        return MergeState(
            memory=memory,
            normalizer=normalizer,
            checkpoints=checkpoints,
            step=0,
        )

    def _token_params(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computa los parámetros de actualización para un token.

        Args:
            x: Tensor [B, model_dim] — embedding del token actual.

        Returns:
            Tuple de:
                decay: [B, S, 1] — factor de decaimiento en (0, 1).
                route: [B, S, 1] — pesos de enrutamiento (distribución sobre slots).
                write_matrix: [B, S, state_dim] — matriz de escritura.
        """
        B = x.size(0)

        # Proyectar y normalizar entrada
        h = rms_norm(self.in_proj(x))  # [B, model_dim]

        # Decay gate: exp(-softplus(·)) → (0, 1)
        decay = torch.exp(-F.softplus(self.decay_proj(h)))  # [B, S]
        decay = decay.unsqueeze(-1)  # [B, S, 1]

        # Route weights: distribución softmax sobre slots
        route = F.softmax(self.route_proj(h), dim=-1)  # [B, S]
        route = route.unsqueeze(-1)  # [B, S, 1]

        # Write: proyección de bajo rango → promediar sobre rango
        write_raw = self.write_proj(h)  # [B, write_rank * state_dim]
        write_raw = write_raw.view(B, self.write_rank, self.state_dim)  # [B, R, Ds]
        write = write_raw.mean(dim=1)  # [B, Ds]

        # Spread por routing: route [B, S, 1] * write [B, 1, Ds] → [B, S, Ds]
        write_matrix = route * write.unsqueeze(1)  # [B, S, Ds]

        return decay, route, write_matrix

    def _update_memory(
        self,
        state: MergeState,
        decay: torch.Tensor,
        route: torch.Tensor,
        write: torch.Tensor,
    ) -> MergeState:
        """Actualiza la memoria con la regla afín.

        Args:
            state: Estado actual de la memoria.
            decay: [B, S, 1] — factor de decaimiento.
            route: [B, S, 1] — pesos de enrutamiento.
            write: [B, S, state_dim] — matriz de escritura.

        Returns:
            Nuevo MergeState con memoria y normalizador actualizados.
            Los checkpoints se preservan sin modificación.
        """
        # Regla afín: M_new = decay * M_old + (1 - decay) * write
        memory_new = decay * state.memory + (1 - decay) * write

        # Actualizar normalizador
        normalizer_new = decay * state.normalizer + (1 - decay) * route

        # Normalizar memoria post-update
        memory_normed = self.post_norm(memory_new)

        return MergeState(
            memory=memory_normed,
            normalizer=normalizer_new,
            checkpoints=state.checkpoints,
            step=state.step,
        )

    def _promote_checkpoint(self, state: MergeState) -> MergeState:
        """Promueve la memoria actual al banco de checkpoints si corresponde.

        Almacena la memoria actual en la posición más reciente del banco
        de checkpoints y desplaza los existentes una posición hacia atrás.
        Solo se ejecuta cuando step es múltiplo de checkpoint_stride.

        Args:
            state: Estado actual después de la actualización de memoria.

        Returns:
            MergeState con checkpoints actualizados (o sin cambios si no toca).
        """
        if self.num_checkpoints <= 0:
            return state

        current_step = state.step
        if current_step % self.checkpoint_stride != 0:
            return state

        # Desplazar checkpoints: mover todo una posición hacia atrás
        # ckpt[:, 1:] = ckpt[:, :-1] y insertar la memoria actual en posición 0
        ckpt = state.checkpoints.clone()
        if self.num_checkpoints > 1:
            ckpt[:, 1:] = ckpt[:, :-1].clone()
        ckpt[:, 0] = state.memory

        return MergeState(
            memory=state.memory,
            normalizer=state.normalizer,
            checkpoints=ckpt,
            step=state.step,
        )

    def read_context(self, x: torch.Tensor, state: MergeState) -> torch.Tensor:
        """Lee información de la memoria usando atención sobre slots.

        Computa atención query-key-value sobre los slots de memoria actuales,
        y opcionalmente también sobre los checkpoints del banco.

        Args:
            x: Tensor [B, model_dim] — embedding del token actual.
            state: Estado de memoria actual.

        Returns:
            Tensor [B, model_dim] — información leída de la memoria.
        """
        # Query desde la entrada
        query = self.query_proj(rms_norm(x))  # [B, state_dim]

        # Keys y values de la memoria actual
        keys = self.key_proj(state.memory)  # [B, S, state_dim]
        vals = self.value_proj(state.memory)  # [B, S, model_dim]

        # Atención sobre slots actuales
        scale = math.sqrt(self.state_dim)
        scores = torch.einsum("bd,bsd->bs", query, keys) / scale  # [B, S]
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, S, 1]
        ctx = (weights * vals).sum(dim=1)  # [B, model_dim]

        # Si hay checkpoints, leer también de ellos
        if self.num_checkpoints > 0:
            # Pesos por nivel fijos (no aprendidos)
            level_logits = torch.linspace(
                0, -1, self.num_checkpoints, device=x.device, dtype=x.dtype
            )
            level_weights = F.softmax(level_logits, dim=0)  # [K]

            ckpt_ctx = torch.zeros_like(ctx)  # [B, model_dim]
            for i in range(self.num_checkpoints):
                # Leer del checkpoint i
                ckpt_mem = state.checkpoints[:, i]  # [B, S, state_dim]
                ckpt_keys = self.key_proj(ckpt_mem)  # [B, S, state_dim]
                ckpt_vals = self.value_proj(ckpt_mem)  # [B, S, model_dim]

                ckpt_scores = torch.einsum("bd,bsd->bs", query, ckpt_keys) / scale
                ckpt_weights = F.softmax(ckpt_scores, dim=-1).unsqueeze(-1)
                ckpt_read = (ckpt_weights * ckpt_vals).sum(dim=1)  # [B, model_dim]
                ckpt_ctx = ckpt_ctx + level_weights[i] * ckpt_read

            # Combinar lectura actual con checkpoints
            ctx = 0.5 * ctx + 0.5 * ckpt_ctx

        return ctx

    def forward(
        self, x: torch.Tensor, state: MergeState
    ) -> Tuple[torch.Tensor, MergeState]:
        """Procesa un token y actualiza el estado de memoria.

        Args:
            x: Tensor [B, model_dim] — embedding del token actual.
            state: MergeState — estado de memoria previo (batched).

        Returns:
            Tuple de (output [B, model_dim], nuevo MergeState).
        """
        # 1. Computar parámetros del token
        decay, route, write = self._token_params(x)

        # 2. Actualizar memoria
        state = self._update_memory(state, decay, route, write)

        # 3. Incrementar contador de paso
        new_step = state.step + 1
        state = MergeState(
            memory=state.memory,
            normalizer=state.normalizer,
            checkpoints=state.checkpoints,
            step=new_step,
        )
        self._step.fill_(new_step)

        # 4. Promover checkpoint si corresponde
        state = self._promote_checkpoint(state)

        # 5. Leer contexto de la memoria
        ctx = self.read_context(x, state)

        # 6. Proyección de salida
        output = self.out_proj(ctx)

        # 7. Puerta residual o suma residual
        if self.residual_gate is not None:
            gate = torch.sigmoid(self.residual_gate(x))
            output = gate * x + (1 - gate) * output
        else:
            output = x + output

        # 8. Dropout
        output = self.dropout_layer(output)

        return output, state

    def forward_sequence(
        self, x: torch.Tensor, state: Optional[MergeState] = None
    ) -> Tuple[torch.Tensor, MergeState]:
        """Procesa una secuencia completa de tokens secuencialmente.

        Args:
            x: Tensor [B, T, model_dim] — secuencia de embeddings.
            state: Estado inicial. Si None, se crea uno nuevo.

        Returns:
            Tuple de (outputs [B, T, model_dim], estado final MergeState).
        """
        B, T, _ = x.shape

        if state is None:
            state = self.init_state(B, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(T):
            token = x[:, t, :]  # [B, model_dim]
            out, state = self.forward(token, state)
            outputs.append(out)

        # Concatenar salidas: lista de [B, model_dim] → [B, T, model_dim]
        outputs = torch.stack(outputs, dim=1)

        return outputs, state

    @staticmethod
    def compose_affine(
        decay1: torch.Tensor,
        write1: torch.Tensor,
        decay2: torch.Tensor,
        write2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compone dos transformaciones afines para prefix-scan paralelo.

        Para dos actualizaciones afines T1(S) = decay1*S + write1 y
        T2(S) = decay2*S + write2, la composición T2(T1(S)) equivale a:
            (decay2 * decay1) * S + (decay2 * write1 + write2)

        Args:
            decay1: Factor de decaimiento de la primera transformación.
            write1: Término de escritura de la primera transformación.
            decay2: Factor de decaimiento de la segunda transformación.
            write2: Término de escritura de la segunda transformación.

        Returns:
            Tuple (decay_composed, write_composed) tal que aplicar el resultado
            es equivalente a aplicar T1 seguido de T2.
        """
        decay_composed = decay2 * decay1
        write_composed = decay2 * write1 + write2
        return decay_composed, write_composed

    def token_update_coefficients(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcula coeficientes afines (decay, write) para un token.

        Útil para construir el scan paralelo durante entrenamiento.
        La actualización afín es: M_new = decay * M_old + effective_write,
        donde effective_write = (1 - decay) * write_matrix.

        Args:
            x: Tensor [B, model_dim].

        Returns:
            Tuple (decay, effective_write):
                decay: [B, S, 1] — coeficientes de decaimiento.
                effective_write: [B, S, state_dim] — término de escritura
                    escalado por (1 - decay).
        """
        decay, _route, write_matrix = self._token_params(x)
        effective_write = (1 - decay) * write_matrix
        return decay, effective_write
