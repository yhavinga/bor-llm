import subprocess
import os

def import_model_to_ollama(model_name: str, base_model: str, system_prompt: str):
    """Import a model into Ollama with custom configuration."""
    modelfile_content = f"""FROM {base_model}

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
PARAMETER num_ctx 8192
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
        if os.path.exists("Modelfile"):
            os.remove("Modelfile")



def main():
    model_name = "bor-dutch"
    base_model = "openhermes"
    system_prompt = "Je bent een behulpzame AI-assistent die vragen beantwoordt in het Nederlands."
    
    import_model_to_ollama(model_name, base_model, system_prompt)

if __name__ == "__main__":
    main()