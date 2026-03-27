import docker
import asyncio
import logging
import time
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PythonSandbox:
    """Isolated Python Docker Sandbox."""
    def __init__(self, image: str = "naminsight-toolbox"):
        self.image = image
        try:
            self.client = docker.from_env()
        except Exception as e:
            logger.error(f"Failed to connect to Docker engine: {str(e)}")
            self.client = None

    async def execute_code(self, code: str, timeout_seconds: int = 15) -> Dict[str, Any]:
        """Async eval wrapper using ThreadPoolExecutor."""
        if not self.client:
            return {
                "output": "",
                "error": "Docker engine not available. Ensure Docker daemon is running.",
                "exit_code": -1,
                "status": "error"
            }
        return await asyncio.to_thread(self._run_in_container, code, timeout_seconds)

    def _run_in_container(self, code: str, timeout_seconds: int) -> Dict[str, Any]:
        container = None
        start_time = time.time()
        try:
            # cache script to tmp file avoiding CLI escapes
            script_path = os.path.join(os.getcwd(), 'exports', 'script.py')
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(code)

            # spawn dynamic container
            host_pwd = os.getenv("HOST_PWD", os.getcwd())
            host_exports = os.path.join(host_pwd, "exports")
            
            container = self.client.containers.run(
                self.image,
                command=["python", "/scratch/script.py"],
                detach=True,
                network_mode="none", # disable network
                mem_limit="512m",    # max 512MB RAM
                user="1000:1000",    # non-root execution
                cpu_quota=50000,     # cap CPU at 0.5
                cap_drop=["ALL"],    # strip OS caps
                read_only=True,      # immutable root FS
                volumes={host_exports: {"bind": "/scratch", "mode": "rw"}}
            )
            
            # async polling timeout loop
            while time.time() - start_time < timeout_seconds:
                container.reload()
                if container.status == "exited":
                    exit_code = container.attrs["State"]["ExitCode"]
                    logs = container.logs(stdout=True, stderr=True).decode('utf-8')
                    return {
                        "output": logs,
                        "error": "Execution failed" if exit_code != 0 else None,
                        "exit_code": exit_code,
                        "status": "success" if exit_code == 0 else "error"
                    }
                time.sleep(0.5)

            # handle sigkill on timeout
            container.kill()
            return {
                "output": "",
                "error": f"Execution timed out after {timeout_seconds} seconds",
                "exit_code": 124, 
                "status": "timeout"
            }
        except Exception as e:
            logger.error(f"Sandbox runtime error: {str(e)}")
            return {
                "output": "",
                "error": str(e),
                "exit_code": -1,
                "status": "error"
            }
        finally:
            if container:
                try:
                    # rm container post-exec
                    container.remove(force=True)
                except Exception:
                    pass

# singleton sandbox instance
sandbox = PythonSandbox()

async def execute_python_code(code: str, timeout: int = 15) -> Dict[str, Any]:
    return await sandbox.execute_code(code, timeout)
