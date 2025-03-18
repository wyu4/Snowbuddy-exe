from cx_Freeze import setup, Executable

required_packages = []
try:
    with open("requirements.txt") as f:
        required_packages = [line.strip() for line in f if line.strip()]
        print(f'Required packages: {str(required_packages)}')
except Exception as e:
    print(f'Could not load requirements.txt: {e}')

setup(
    name="StartServer",
    version="0.1.0",
    description="SnowBuddy is a Chrome extension that analyzes and highlights offensive sentences in Gmail. It acts as an inverse preventive tool, sometimes making you think you are being offensive when you are not, or vice versa.",
    options={
        "build_exe": {
            "packages": required_packages,
            "include_files": [
                "python\\snowflake_classifier\\config.json",
                "python\\snowflake_classifier\\model.safetensors",
                "python\\snowflake_classifier\\special_tokens_map.json",
                "python\\snowflake_classifier\\tokenizer_config.json",
                "python\\snowflake_classifier\\vocab.txt"
            ],
            "optimize": 0,
        }
    },
    executables=[Executable("python\\main.py", icon="snowbuddy.ico", base=None)],
)