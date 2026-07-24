"""Configuración para la arquitectura Causal Matrix Merge.

Este módulo define el dataclass inmutable CausalMatrixMergeConfig que centraliza
todos los hiperparámetros de la arquitectura. Soporta serialización a diccionario
y reconstrucción desde diccionario para persistencia.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass(frozen=True)
class CausalMatrixMergeConfig:
    """Configuración completa para la arquitectura Causal Matrix Merge.

    Define todos los hiperparámetros necesarios para instanciar un modelo
    Causal Matrix Merge. Es inmutable (frozen) para garantizar consistencia
    una vez creada.

    Attributes:
        vocab_size: Tamaño del vocabulario de tokens.
            Tipo: int. Rango válido: >= 1. Default: 1000.
        model_dim: Dimensión del embedding y residual stream.
            Tipo: int. Rango válido: >= 1. Default: 128.
        state_dim: Dimensión de cada slot de memoria.
            Tipo: int. Rango válido: >= 1. Default: 64.
        num_slots: Número de slots de memoria independientes.
            Tipo: int. Rango válido: >= 1. Default: 8.
        num_layers: Número de capas CausalMatrixMerge apiladas.
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
    """

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

    def __post_init__(self) -> None:
        """Valida que todos los campos tengan valores dentro de rangos permitidos.

        Raises:
            ValueError: Si algún campo tiene un valor inválido.
        """
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
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError("dropout debe estar en [0, 1]")
        if self.max_context < 1:
            raise ValueError("max_context debe ser ≥ 1")

    def to_dict(self) -> Dict[str, Any]:
        """Serializa la configuración a diccionario.

        Returns:
            Diccionario con todos los campos y sus valores actuales.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CausalMatrixMergeConfig":
        """Reconstruye una configuración desde un diccionario.

        Args:
            d: Diccionario con los campos de configuración. Las claves deben
               coincidir con los nombres de los atributos del dataclass.

        Returns:
            Nueva instancia de CausalMatrixMergeConfig con los valores del dict.

        Raises:
            TypeError: Si el diccionario contiene claves no reconocidas.
            ValueError: Si algún valor no cumple las validaciones.
        """
        return cls(**d)
