"""Estado inmutable de memoria para la arquitectura Causal Matrix Merge.

Este módulo define MergeState, la estructura de datos que representa el contexto
comprimido de toda la historia de la secuencia. El estado es de tamaño fijo
independientemente de cuántos tokens se hayan procesado — esto es lo que habilita
contexto infinito. La memoria nunca crece con la longitud de la secuencia; solo
mejora su contenido informacional.
"""

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class MergeState:
    """Estado inmutable de una capa CausalMatrixMerge.

    Representa el contexto comprimido de toda la historia de la secuencia.
    Es de tamaño fijo sin importar cuántos tokens se hayan procesado, lo que
    habilita contexto infinito. Al ser un dataclass inmutable (frozen), facilita
    la inferencia funcional token-a-token y la serialización segura.

    Attributes:
        memory: Tensor [slots, state_dim] — memoria matricial actual que almacena
            la información acumulada de los tokens procesados.
        normalizer: Tensor [slots, 1] — normalizador acumulado para estabilizar
            las actualizaciones afines.
        checkpoints: Tensor [num_checkpoints, slots, state_dim] — banco de
            checkpoints que preserva estados históricos para lectura de largo alcance.
        step: int — contador de pasos procesados, usado para determinar cuándo
            almacenar un nuevo checkpoint.
    """

    memory: torch.Tensor       # [slots, state_dim]
    normalizer: torch.Tensor   # [slots, 1]
    checkpoints: torch.Tensor  # [num_checkpoints, slots, state_dim]
    step: int

    @classmethod
    def initial(cls, num_slots: int, state_dim: int, num_checkpoints: int) -> "MergeState":
        """Crea estado inicial con memoria en ceros y normalizador en unos.

        Args:
            num_slots: Número de slots de memoria.
            state_dim: Dimensión de cada slot de memoria.
            num_checkpoints: Profundidad del banco de checkpoints.

        Returns:
            MergeState con memory y checkpoints inicializados a cero,
            normalizer inicializado a uno, y step en 0.
        """
        return cls(
            memory=torch.zeros(num_slots, state_dim),
            normalizer=torch.ones(num_slots, 1),
            checkpoints=torch.zeros(num_checkpoints, num_slots, state_dim),
            step=0,
        )

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Serializa el estado a un diccionario de tensores.

        El campo step se convierte a tensor para compatibilidad con
        torch.save/torch.load y consistencia en el formato de serialización.

        Returns:
            Diccionario con claves 'memory', 'normalizer', 'checkpoints' y 'step',
            donde todos los valores son tensores de PyTorch.
        """
        return {
            "memory": self.memory,
            "normalizer": self.normalizer,
            "checkpoints": self.checkpoints,
            "step": torch.tensor(self.step),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, torch.Tensor]) -> "MergeState":
        """Reconstruye un MergeState desde un diccionario de tensores.

        Args:
            d: Diccionario con claves 'memory', 'normalizer', 'checkpoints' y
                'step'. El campo 'step' debe ser un tensor escalar que se
                convertirá de vuelta a int.

        Returns:
            MergeState reconstruido con los valores del diccionario.
        """
        return cls(
            memory=d["memory"],
            normalizer=d["normalizer"],
            checkpoints=d["checkpoints"],
            step=int(d["step"].item()),
        )
