"""
本包在整个项目中的角色：对外暴露“HTTP 服务（FastAPI app）”的稳定入口。

为什么这个文件存在？
- 让启动脚本/ASGI Server 可以稳定地通过 `from service import app` 拿到 FastAPI 实例；
  不需要关心具体实现文件名（`service.service`）。
- 这是典型的“包级 API 门面（facade）”：内部结构可调整，对外导入路径保持稳定。

典型调用者：
- `src/run_service.py` 或 `uvicorn service:app` 这类启动方式。
- 测试用例（`tests/service/*`）在需要导入 app 进行 TestClient/AsyncClient 测试时。
"""

from service.service import app

__all__ = ["app"]
