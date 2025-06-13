# docker_orchestrator.py

import os
import docker  # Import the Docker library
import tempfile
import shutil
import logging

logger = logging.getLogger(__name__) # This links to the root logger potentially configured elsewhere

def is_docker_running():
    try:
        client = docker.from_env()
        client.ping()  # Checks if the Docker daemon is reachable
        return True
    except docker.errors.APIError:
        return False
    except docker.errors.DockerException:
        # This handles cases where Docker is not installed or misconfigured
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False  # Handle other potential errors


def run_deobfuscation_in_docker(llm_python_code, dockerfile_content, original_file_path):
    client = docker.from_env()
    tmpdir = tempfile.mkdtemp()
    docker_error = ""
    # Write both the Python script and Dockerfile to the temporary directory.
    python_file_path = os.path.join(tmpdir, "decoder.py")
    with open(python_file_path, "w") as f:
        f.write(llm_python_code)

    dockerfile_path = os.path.join(tmpdir, "Dockerfile")
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)

    print("\n\nBuilding Docker image...")
    try:
        image, build_logs = client.images.build(path=tmpdir, dockerfile="Dockerfile", tag="deobfuscator-image", rm=True)
        for line in build_logs:
            if "stream" in line:
                print(line["stream"].strip())

        container = client.containers.run(
            'deobfuscator-image:latest',
            detach=True,
        )

        exit_code = container.wait(timeout=60)
        logs = container.logs(stdout=True, stderr=True)
        output = logs.decode()
        print(f"Container exited with code {exit_code}")

        if exit_code != 0:
            print(f"Docker container finished successfully. \nContainer Logs:\n{output}")
            return output, exit_code
        else:
            return None, exit_code  # Return output even if it's an empty string

    except docker.errors.BuildError as e:
        print(f"Docker build error: {e}")
        docker_error = f"Docker build error: {e.msg}"
    except docker.errors.ContainerError as e:
        print(f"Docker container error: {e}")
        docker_error = f"Docker container error: {e.stderr.decode()}"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        docker_error = f"An unexpected error occurred: {e}"
    finally:
        #  Automatic Container Cleanup 
        if container is not None:
            try:
                print("Cleaning up Docker container...")
                container.remove(force=True)  # Force remove in case it's stuck
                logger.info(f"Successfully removed Docker container: {container.id}")
                print("Docker container removed.")
            except docker.errors.NotFound:
                logger.warning(f"Container with ID {container.id} not found during cleanup (already removed?).")
                print("Container already removed or not found during cleanup.")
            except Exception as e:
                logger.error(f"Error removing Docker container {container.id}: {e}", exc_info=True)
                print(f"Error removing Docker container: {e}")
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
    return docker_error, exit_code