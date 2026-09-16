"""工具纪律测试：不抛异常、失败也是结果、精确字符串编辑。"""
import sys

from xianyu_agent.cancel import CancellationToken
from xianyu_agent.tools import ToolContext, builtin_tools


def make_tctx():
    return ToolContext(cancel=CancellationToken())


def test_read_file_missing_returns_error_string(tmp_path):
    tool = {t.name: t for t in builtin_tools()}["read_file"]
    result = tool.execute({"path": str(tmp_path / "nope.txt")}, make_tctx())
    assert result.startswith("error:")


def test_write_then_read(tmp_path):
    tools = {t.name: t for t in builtin_tools()}
    p = tmp_path / "hello.txt"
    assert "已写入" in tools["write_file"].execute(
        {"path": str(p), "content": "hello world"}, make_tctx())
    assert tools["read_file"].execute({"path": str(p)}, make_tctx()) == "hello world"


def test_edit_requires_unique_match(tmp_path):
    tools = {t.name: t for t in builtin_tools()}
    p = tmp_path / "dup.txt"
    p.write_text("aa b aa", encoding="utf-8")

    # 出现两次：拒绝动手
    result = tools["edit"].execute(
        {"path": str(p), "old_string": "aa", "new_string": "cc"}, make_tctx())
    assert result.startswith("error:") and "2 次" in result
    assert p.read_text(encoding="utf-8") == "aa b aa"      # 文件没被改坏

    # 唯一匹配：替换成功
    result = tools["edit"].execute(
        {"path": str(p), "old_string": "b", "new_string": "c"}, make_tctx())
    assert "已替换" in result
    assert p.read_text(encoding="utf-8") == "aa c aa"

    # 找不到：报错
    result = tools["edit"].execute(
        {"path": str(p), "old_string": "zzz", "new_string": "x"}, make_tctx())
    assert result.startswith("error:")


def test_run_bash_failure_is_result_not_exception():
    tool = {t.name: t for t in builtin_tools()}["run_bash"]
    result = tool.execute({"command": f'"{sys.executable}" -c "import sys; sys.exit(1)"'},
                          make_tctx())
    assert result.startswith("exit=1")     # 退出码和输出一起返回，不抛异常
