"""Install the Dory model into a pinned mlc-llm checkout: python apply.py <mlc-llm dir>"""
import pathlib
import shutil
import sys

here = pathlib.Path(__file__).parent
root = pathlib.Path(sys.argv[1]) / "python" / "mlc_llm"

pkg = root / "model" / "dory"
pkg.mkdir(exist_ok=True)
(pkg / "__init__.py").write_text("")
for name in ("dory_model.py", "dory_loader.py"):
    shutil.copy(here / name, pkg / name)

shutil.copy(here / "dory_template.py", root / "conversation_template" / "dory.py")

p = root / "conversation_template" / "__init__.py"
s = p.read_text()
if "    dory,\n" not in s:
    assert "    dolly,\n" in s
    s = s.replace("    dolly,\n", "    dolly,\n    dory,\n", 1)
    p.write_text(s)

p = root / "model" / "model.py"
s = p.read_text()
if "dory_model" not in s:
    anchor = "from .gemma import gemma_loader, gemma_model\n"
    assert anchor in s
    s = s.replace(anchor, "from .dory import dory_loader, dory_model\n" + anchor, 1)
    entry = '''    "dory": Model(
        name="dory",
        model=dory_model.DoryForCausalLM,
        config=dory_model.DoryConfig,
        source={
            "huggingface-torch": dory_loader.huggingface,
            "huggingface-safetensor": dory_loader.huggingface,
        },
        quantize=make_quantization_functions(
            dory_model.DoryForCausalLM,
            supports_ft_quant=False,
        ),
    ),
'''
    anchor = '    "gemma3": Model(\n'
    assert anchor in s
    s = s.replace(anchor, entry + anchor, 1)
    p.write_text(s)

p = root / "interface" / "gen_config.py"
s = p.read_text()
if '"dory",' not in s:
    anchor = '    "qwen3",\n'
    assert anchor in s
    s = s.replace(anchor, '    "dory",\n' + anchor, 1)
    p.write_text(s)
print("dory installed")
