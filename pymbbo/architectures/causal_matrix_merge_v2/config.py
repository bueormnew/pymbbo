"""Configuración para la arquitectura Causal Matrix Merge v2.

Este módulo define el dataclass inmutable CausalMatrixMergeV2Config que centraliza
todos los hiperparámetros de la arquitectura v2. Incluye todos los campos de v1
más los nuevos campos para SwiGLU MLP, Sparse Routing, Adaptive Merge,
Learned Checkpoint Selection y Per-Slot Local Merge Improvements.
Soporta serialización a diccionario y reconstrucción desde diccionario para persistencia.
"""

from dataclasses import dataclass, asdict, fields
from typing import Any, Dict


@dataclass(frozen=True)
class CausalMatrixMergeV2Config:
    """Configuración completa para la arquitectura Causal Matrix Merge v2.

    Define todos los hiperparámetros necesarios para instanciar un modelo
    Causal Matrix Merge v2. Es inmutable (frozen) para garantizar consistencia
    una vez creada. Incluye todos los campos de v1 sin eliminar ni renombrar
    ninguno, más campos nuevos para los módulos añadidos en v2.

    Attributes:
        vocab_size: Tamaño del vocabulario de tokens.
            Tipo: int. Rango válido: >= 1. Default: 1000.
        model_dim: Dimensión del embedding y residual stream.
            Tipo: int. Rango válido: >= 1. Default: 128.
        state_dim: Dimensión de cada slot de memoria.
            Tipo: int. Rango válido: >= 1. Default: 64.
        num_slots: Número de slots de memoria independientes.
            Tipo: int. Rango válido: >= 1. Default: 8.
        num_layers: Número de capas CausalMatrixMergeLayerV2 apiladas.
            Tipo: int. Rango válido: >= 1. Default: 2.
        num_checkpoints: Profundidad del banco de checkpoints para lectura
            de largo alcance.
            Tipo: int. Rango válido: >= 1. Default: 4.
        checkpoint_stride: Cada cuántos pasos se guarda un checkpoint en el
            banco de memoria.
            Tipo: int. Rango válido: >= 1. Default: 16.
        write_rank: Rango de la proyección de escritura de bajo rango.
            Tipo: int. Rango válido: >= 1. Default: 4.
        dropout: Tasa de dropout aplicada durante entrenamiento.
            Tipo: float. Rango válido: [0.0, 1.0]. Default: 0.1.
        use_residual_gate: Si se usa puerta residual aprendida para mezclar
            la salida de lectura con la entrada original.
            Tipo: bool. Default: True.
        max_context: Longitud máxima de contexto soportada por el modelo.
            Tipo: int. Rango válido: >= 1. Default: 2048.
        ffn_mult: Factor multiplicador de la dimensión interna del Sub_Bloque_MLP
            respecto a model_dim (dimensión oculta = model_dim * ffn_mult).
            Tipo: float. Rango válido: > 0. Default: 2.667.
        top_k_slots: Número de slots seleccionados por el Routing Sparse para
            escritura por token. Debe ser <= num_slots.
            Tipo: int. Rango válido: [1, num_slots]. Default: 4.
        use_adaptive_merge: Si se habilita el Merge Adaptativo que modula
            el decay en función del contenido de memoria.
            Tipo: bool. Default: True.
        use_learned_checkpoints: Si se habilita el Checkpoint Selector aprendido
            que reemplaza los pesos fijos de v1.
            Tipo: bool. Default: True.
        use_per_slot_decay: Si se habilita el decay adaptativo local per-slot.
            Cuando es True (y use_adaptive_merge también es True), cada slot
            computa su propio factor de modulación de decay usando exclusivamente
            el token actual y su contenido individual, sin acceder a otros slots.
            Cuando es False, se usa el mecanismo de decay adaptativo global
            existente (con memory.mean) como fallback.
            Tipo: bool. Default: True.
        use_per_slot_write: Si se habilita la adaptación per-slot del vector
            de escritura mediante FiLM. Cuando es True, cada slot recibe una
            versión ligeramente diferente del write condicionada por su contenido
            actual. Cuando es False, se distribuye el vector de escritura base
            de forma uniforme a todos los slots (comportamiento original).
            Tipo: bool. Default: True.
    """

    # === Campos heredados de v1 (mismos defaults) ===
    vocab_size: int = 1000
    model_dim: int = 128
    state_dim: int = 64
    num_slots: int = 8
    num_layers: int = 2
    num_checkpoints: int = 4
    checkpoint_stride: int = 16
    write_rank: int = 4
    dropout: float = 0.1
    use_residual_gate: bool = True
    max_context: int = 2048

    # === Campos nuevos de v2 ===
    ffn_mult: float = 2.667
    top_k_slots: int = 4
    use_adaptive_merge: bool = True
    use_learned_checkpoints: bool = True

    # === Campos per-slot local merge ===
    use_per_slot_decay: bool = True
    use_per_slot_write: bool = True

    def __post_init__(self) -> None:
        """Valida que todos los campos tengan valores dentro de rangos permitidos.

        Raises:
            ValueError: Si algún campo tiene un valor inválido.
        """
        # Validaciones de campos enteros dimensionales (>= 1)
        if self.vocab_size < 1:
            raise ValueError("vocab_size debe ser ≥ 1")
        if self.model_dim < 1:
            raise ValueError("model_dim debe ser ≥ 1")
        if self.state_dim < 1:
            raise ValueError("state_dim debe ser ≥ 1")
        if self.num_slots < 1:
            raise ValueError("num_slots debe ser ≥ 1")
        if self.num_layers < 1:
            raise ValueError("num_layers debe ser ≥ 1")
        if self.num_checkpoints < 1:
            raise ValueError("num_checkpoints debe ser ≥ 1")
        if self.checkpoint_stride < 1:
            raise ValueError("checkpoint_stride debe ser ≥ 1")
        if self.write_rank < 1:
            raise ValueError("write_rank debe ser ≥ 1")
        if self.max_context < 1:
            raise ValueError("max_context debe ser ≥ 1")

        # Validación de dropout
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError("dropout debe estar en [0, 1]")

        # Validaciones de campos nuevos de v2
        if self.ffn_mult <= 0:
            raise ValueError("ffn_mult debe ser > 0")
        if not (1 <= self.top_k_slots <= self.num_slots):
            raise ValueError(
                f"top_k_slots debe estar en [1, num_slots={self.num_slots}], "
                f"pero se recibió {self.top_k_slots}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serializa la configuración a diccionario.

        Returns:
            Diccionario con todos los campos y sus valores actuales.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CausalMatrixMergeV2Config":
        """Reconstruye una configuración desde un diccionario.

        Soporta backward compatibility (dicts antiguos sin campos nuevos usan
        los defaults del dataclass) y forward compatibility (claves
        desconocidas se filtran silenciosamente).

        Args:
            d: Diccionario con los campos de configuración. Las claves deben
               coincidir con los nombres de los atributos del dataclass.
               Claves extra (no reconocidas) se ignoran. Claves faltantes
               usan el default del campo correspondiente.

        Returns:
            Nueva instancia de CausalMatrixMergeV2Config con los valores del dict.

        Raises:
            ValueError: Si algún valor no cumple las validaciones.
        """
        known_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)
