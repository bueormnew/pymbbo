"""Modelo de lenguaje autoregresivo Causal Matrix Merge.

Este módulo implementa CausalMatrixMergeModel, el modelo de lenguaje completo
que apila múltiples capas CausalMatrixMerge con token embedding y cabeza de logits.
NO es un transformer — utiliza memoria matricial de tamaño fijo para comprimir
contexto infinito mediante la regla de actualización afín.

El modelo es compatible con el ecosistema pymbbo (build_model, save_model, fit, etc.)
mediante la herencia de BaseArchitecture y el decorador @register_architecture.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from pymbbo.architectures.base_arch import BaseArchitecture
from pymbbo.models.registry import register_architecture

from .config import CausalMatrixMergeConfig
from .generation import sample_token
from .merge import CausalMatrixMerge
from .state import MergeState


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization con escala aprendida.

    Normaliza el tensor de entrada usando la raíz cuadrática media y aplica
    una escala aprendida por dimensión.

    Args:
        dim: Dimensión de la última dimensión del tensor de entrada.
        eps: Epsilon para estabilidad numérica.
    """

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aplica RMS normalization con escala aprendida.

        Args:
            x: Tensor de forma [..., dim].

        Returns:
            Tensor normalizado con la misma forma que x.
        """
        rms = x.pow(2).mean(dim=-1, keepdim=True).clamp_min(self.eps).rsqrt()
        return x * rms * self.weight


@register_architecture("causal_matrix_merge")
class CausalMatrixMergeModel(BaseArchitecture):
    """Modelo de lenguaje autoregresivo basado en Causal Matrix Merge.

    Apila N capas CausalMatrixMerge con token embedding y cabeza de logits.
    La memoria es fija y comprime contexto infinito mediante la regla de merge.
    Compatible con el ecosistema pymbbo (build_model, save_model, fit, etc.).

    En modo entrenamiento (forward()), los estados se reinician en cada llamada
    para que cada secuencia del batch sea independiente. En modo generación
    (_forward_with_state()), los estados se preservan entre llamadas.

    Attributes:
        ARCH_NAME: Nombre de la arquitectura para el registro de pymbbo.
        token_embedding: Capa de embedding de tokens.
        layers: Lista de capas CausalMatrixMerge apiladas.
        final_norm: Normalización RMS antes de la cabeza de logits.
        lm_head: Cabeza lineal que produce logits sobre el vocabulario.
    """

    ARCH_NAME = "causal_matrix_merge"

    def __init__(
        self,
        vocab_size: int = 1000,
        model_dim: int = 128,
        state_dim: int = 64,
        num_slots: int = 8,
        num_layers: int = 2,
        num_checkpoints: int = 4,
        checkpoint_stride: int = 16,
        write_rank: int = 4,
        dropout: float = 0.1,
        use_residual_gate: bool = True,
        max_context: int = 2048,
        **kwargs,
    ) -> None:
        """Inicializa el modelo CausalMatrixMerge.

        Args:
            vocab_size: Tamaño del vocabulario de tokens.
            model_dim: Dimensión del embedding y residual stream.
            state_dim: Dimensión de cada slot de memoria.
            num_slots: Número de slots de memoria independientes.
            num_layers: Número de capas CausalMatrixMerge apiladas.
            num_checkpoints: Profundidad del banco de checkpoints.
            checkpoint_stride: Cada cuántos pasos se guarda un checkpoint.
            write_rank: Rango de la proyección de escritura de bajo rango.
            dropout: Tasa de dropout aplicada durante entrenamiento.
            use_residual_gate: Si se usa puerta residual aprendida.
            max_context: Longitud máxima de contexto soportada.
            **kwargs: Argumentos adicionales pasados a BaseArchitecture.
        """
        # Pasar todos los params a super para almacenar en config_kwargs
        super().__init__(
            vocab_size=vocab_size,
            model_dim=model_dim,
            state_dim=state_dim,
            num_slots=num_slots,
            num_layers=num_layers,
            num_checkpoints=num_checkpoints,
            checkpoint_stride=checkpoint_stride,
            write_rank=write_rank,
            dropout=dropout,
            use_residual_gate=use_residual_gate,
            max_context=max_context,
            **kwargs,
        )

        # Crear configuración
        self.config = CausalMatrixMergeConfig(
            vocab_size=vocab_size,
            model_dim=model_dim,
            state_dim=state_dim,
            num_slots=num_slots,
            num_layers=num_layers,
            num_checkpoints=num_checkpoints,
            checkpoint_stride=checkpoint_stride,
            write_rank=write_rank,
            dropout=dropout,
            use_residual_gate=use_residual_gate,
            max_context=max_context,
        )

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, model_dim)

        # Capas CausalMatrixMerge apiladas
        self.layers = nn.ModuleList(
            [CausalMatrixMerge(self.config) for _ in range(num_layers)]
        )

        # Normalización final antes de LM head
        self.final_norm = RMSNorm(model_dim)

        # Cabeza de logits (sin bias, weight tying opcional)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)

        # Estados internos de memoria (uno por capa)
        self._states: List[Optional[MergeState]] = [None] * num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass para entrenamiento — estados se reinician cada llamada.

        Procesa una secuencia completa de token IDs a través de todas las capas.
        Al inicio de cada llamada, los estados se reinician para que cada
        secuencia del batch sea independiente (compatible con entrenamiento).

        Args:
            x: Tensor [batch, seq_len] de token IDs enteros en [0, vocab_size).

        Returns:
            Logits [batch, seq_len, vocab_size] donde logits[:, t, :] predice
            el token en posición t+1.
        """
        B, T = x.shape

        # Reiniciar estados (cada secuencia de entrenamiento empieza limpia)
        self.reset_state()

        # Embedding de tokens
        h = self.token_embedding(x)  # [B, T, model_dim]

        # Procesar a través de cada capa
        for i, layer in enumerate(self.layers):
            state = layer.init_state(B, device=x.device, dtype=h.dtype)
            h, new_state = layer.forward_sequence(h, state)
            self._states[i] = new_state

        # Normalización final
        h = self.final_norm(h)  # [B, T, model_dim]

        # Cabeza de logits
        logits = self.lm_head(h)  # [B, T, vocab_size]

        return logits

    def reset_state(self) -> None:
        """Reinicia el estado de memoria de todas las capas a None.

        Después de llamar a este método, la próxima invocación de forward o
        _forward_with_state creará estados frescos (memoria en ceros).
        """
        self._states = [None] * len(self.layers)

    def get_state(self) -> List[Dict[str, torch.Tensor]]:
        """Retorna el estado de memoria de todas las capas como lista de dicts.

        Para capas cuyo estado es None, retorna un estado inicial con las
        dimensiones correspondientes a la configuración.

        Returns:
            Lista de diccionarios, uno por capa, cada uno con claves
            'memory', 'normalizer', 'checkpoints' y 'step'.
        """
        result = []
        for state in self._states:
            if state is None:
                # Crear estado inicial con dimensiones de la config
                initial = MergeState.initial(
                    num_slots=self.config.num_slots,
                    state_dim=self.config.state_dim,
                    num_checkpoints=self.config.num_checkpoints,
                )
                result.append(initial.to_dict())
            else:
                result.append(state.to_dict())
        return result

    def set_state(self, states: List[Dict[str, torch.Tensor]]) -> None:
        """Restaura el estado de memoria de todas las capas desde lista de dicts.

        Args:
            states: Lista de diccionarios (uno por capa) con claves
                'memory', 'normalizer', 'checkpoints' y 'step', como los
                producidos por get_state().

        Raises:
            ValueError: Si la longitud de states no coincide con num_layers.
        """
        if len(states) != len(self.layers):
            raise ValueError(
                f"Se esperan {len(self.layers)} estados, se recibieron {len(states)}"
            )
        self._states = [MergeState.from_dict(s) for s in states]

    def get_config(self) -> Dict[str, Any]:
        """Retorna todos los hiperparámetros para reconstrucción del modelo.

        El diccionario retornado contiene todos los argumentos necesarios para
        instanciar un modelo equivalente con CausalMatrixMergeModel(**config).

        Returns:
            Diccionario con vocab_size, model_dim, state_dim, num_slots,
            num_layers, num_checkpoints, checkpoint_stride, write_rank,
            dropout, use_residual_gate y max_context.
        """
        return self.config_kwargs

    def _forward_with_state(
        self,
        x: torch.Tensor,
        states: Optional[List[Optional[MergeState]]] = None,
    ) -> Tuple[torch.Tensor, List[MergeState]]:
        """Forward pass preservando estados — para generación autoregresiva.

        A diferencia de forward(), este método no reinicia los estados sino que
        los preserva y retorna actualizados para la siguiente llamada.

        Args:
            x: Tensor [batch, seq_len] de token IDs.
            states: Lista de MergeState (uno por capa). Si None o contiene None,
                se crean estados frescos para las capas correspondientes.

        Returns:
            Tuple de:
                logits: [batch, seq_len, vocab_size]
                updated_states: Lista de MergeState actualizados (uno por capa).
        """
        B, T = x.shape

        # Embedding de tokens
        h = self.token_embedding(x)  # [B, T, model_dim]

        # Preparar estados
        if states is None:
            states = [None] * len(self.layers)

        updated_states: List[MergeState] = []

        # Procesar a través de cada capa
        for i, layer in enumerate(self.layers):
            state = states[i]
            if state is None:
                state = layer.init_state(B, device=x.device, dtype=h.dtype)
            h, new_state = layer.forward_sequence(h, state)
            updated_states.append(new_state)

        # Normalización final
        h = self.final_norm(h)  # [B, T, model_dim]

        # Cabeza de logits
        logits = self.lm_head(h)  # [B, T, vocab_size]

        return logits, updated_states

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Genera tokens autoregresivamente a partir de un prompt.

        Procesa el prompt completo a través de _forward_with_state para
        construir el estado de memoria comprimido, y luego genera tokens
        uno a uno alimentando cada nuevo token al modelo. La memoria de
        tamaño fijo se preserva entre llamadas — O(1) por token generado.

        Args:
            prompt_ids: Tensor [batch, seq_len] de token IDs del prompt.
                Puede tener seq_len=0 para generar desde un estado fresco.
            max_new_tokens: Número máximo de tokens nuevos a generar.
                Si es 0, retorna prompt_ids sin cambios.
            temperature: Factor de temperatura para el muestreo. Debe ser > 0.
            top_k: Si se proporciona, filtra a los top-k logits más altos.
                Debe ser >= 1.
            top_p: Si se proporciona, aplica muestreo nucleus con umbral p.
                Debe estar en (0, 1].

        Returns:
            Tensor [batch, seq_len + max_new_tokens] con el prompt original
            concatenado con los tokens generados.

        Raises:
            ValueError: Si temperature <= 0, top_k < 1, o top_p fuera de (0, 1].
        """
        # Validar inputs
        if temperature <= 0:
            raise ValueError("temperature debe ser > 0")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k debe ser >= 1")
        if top_p is not None and (top_p <= 0 or top_p > 1):
            raise ValueError("top_p debe estar en (0, 1]")

        # Edge case: nada que generar
        if max_new_tokens == 0:
            return prompt_ids

        self.eval()

        with torch.no_grad():
            B = prompt_ids.size(0)
            states: Optional[List[Optional[MergeState]]] = None

            # Procesar el prompt completo para construir el estado de memoria
            if prompt_ids.size(1) > 0:
                logits, states = self._forward_with_state(prompt_ids, states)
                # Tomar el último token generado por el prompt como semilla
                last_logits = logits[:, -1, :]  # [B, vocab_size]
            else:
                # Prompt vacío: generar desde estado fresco
                # Necesitamos un token semilla — usar logits desde estado inicial
                # Crear un token dummy (0) para obtener los primeros logits
                dummy = torch.zeros(B, 1, dtype=torch.long, device=prompt_ids.device)
                logits, states = self._forward_with_state(dummy, states)
                last_logits = logits[:, -1, :]  # [B, vocab_size]

            # Muestrear el primer token nuevo
            generated_tokens = []

            # Loop de generación autoregresiva
            for _ in range(max_new_tokens):
                # Muestrear token de los logits actuales
                next_token = sample_token(
                    last_logits, temperature=temperature, top_k=top_k, top_p=top_p
                )  # [B, 1]
                generated_tokens.append(next_token)

                # Alimentar el token generado al modelo (seq_len=1)
                logits, states = self._forward_with_state(next_token, states)
                last_logits = logits[:, -1, :]  # [B, vocab_size]

            # Concatenar tokens generados
            generated = torch.cat(generated_tokens, dim=1)  # [B, max_new_tokens]

            # Retornar prompt + generado
            return torch.cat([prompt_ids, generated], dim=1)
