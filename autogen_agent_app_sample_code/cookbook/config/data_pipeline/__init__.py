from cookbook.config import SerializableConfig, serializable_config_to_yaml
import yaml
from cookbook.config import (
    load_serializable_config_from_yaml,
)
from cookbook.config.data_pipeline.data_pipeline_output import DataPipelineOuputConfig
from cookbook.config.data_pipeline.recursive_text_splitter import (
    RecursiveTextSplitterChunkingConfig,
)
from cookbook.config.data_pipeline.uc_volume_source import UCVolumeSourceConfig


from typing import Any, Dict


class DataPipelineConfig(SerializableConfig):
    source: UCVolumeSourceConfig
    output: DataPipelineOuputConfig
    chunking_config: RecursiveTextSplitterChunkingConfig

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to exclude name and description fields.

        Returns:
            Dict[str, Any]: Dictionary representation of the model excluding name and description.
        """
        model_dumped = super().model_dump(**kwargs)
        model_dumped["source"] = yaml.safe_load(
            serializable_config_to_yaml(self.source)
        )
        model_dumped["output"] = yaml.safe_load(
            serializable_config_to_yaml(self.output)
        )
        model_dumped["chunking_config"] = yaml.safe_load(
            serializable_config_to_yaml(self.chunking_config)
        )
        return model_dumped

    @classmethod
    def _load_class_from_dict(
        cls, class_object, data: Dict[str, Any]
    ) -> "SerializableConfig":
        # Deserialize sub-configs
        data["source"] = load_serializable_config_from_yaml(yaml.dump(data["source"]))
        data["output"] = load_serializable_config_from_yaml(yaml.dump(data["output"]))
        data["chunking_config"] = load_serializable_config_from_yaml(
            yaml.dump(data["chunking_config"])
        )
        return class_object(**data)
