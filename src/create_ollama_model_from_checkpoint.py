import argparse
import subprocess
from pathlib import Path


def import_model_to_ollama(model_name: str, model_path: str, system_prompt: str):
    """Import a model into Ollama with custom configuration."""
    checkpoint_path = Path(model_path)
    gguf_file = checkpoint_path / "converted.gguf"

    if not gguf_file.exists():
        raise FileNotFoundError(f"GGUF file not found at {gguf_file}")

    absolute_model_path = str(gguf_file.absolute())
    modelfile_content = f"""FROM {absolute_model_path}

SYSTEM {system_prompt}

TEMPLATE \"\"\"{{{{- if .System }}}}
<|system|>
{{{{ .System }}}}
</s>
{{{{- end }}}}
<|user|>
{{{{ .Prompt }}}}
</s>
<|assistant|>
\"\"\"

PARAMETER temperature 1.0
PARAMETER num_ctx 4096
PARAMETER stop "<|system|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
PARAMETER stop "</s>"
"""

    # Write Modelfile
    with open("Modelfile", "w") as f:
        f.write(modelfile_content)

    # Create model using Ollama CLI
    try:
        subprocess.run(["ollama", "create", model_name, "-f", "Modelfile"], check=True)
        print(f"Successfully imported model: {model_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating model: {e}")
    finally:
        # Cleanup Modelfile
        # if os.path.exists("Modelfile"):
        #     os.remove("Modelfile")
        ...


def main():
    parser = argparse.ArgumentParser(
        description="Import a model into Ollama from a checkpoint"
    )
    parser.add_argument(
        "model_name", type=str, help="Name for the Ollama model (e.g., bor-chat-qlora)"
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to the checkpoint directory containing safetensors files",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="Je bent een behulpzame AI-assistent die vragen beantwoordt in het Nederlands.",
        help="System prompt for the model",
    )

    args = parser.parse_args()

    import_model_to_ollama(args.model_name, args.checkpoint_dir, args.system_prompt)


if __name__ == "__main__":
    main()
