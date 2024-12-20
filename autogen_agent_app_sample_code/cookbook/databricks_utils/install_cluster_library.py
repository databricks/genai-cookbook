from typing import List

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.compute import (
    Library,
    LibraryFullStatus,
    LibraryInstallStatus,
    PythonPyPiLibrary,
)
import time


def parse_requirements(requirements_path: str) -> List[str]:
    """Parse requirements.txt file and return list of package specifications."""
    packages = []
    with open(requirements_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                packages.append(line)
    return packages


def wait_for_library_installation(
    w: WorkspaceClient, cluster_id: str, timeout_minutes: int = 20
):
    """Wait for all libraries to be installed or fail."""
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    final_states = {
        LibraryInstallStatus.INSTALLED,
        LibraryInstallStatus.FAILED,
        LibraryInstallStatus.SKIPPED,
    }

    while True:
        if time.time() - start_time > timeout_seconds:
            print(
                f"Timeout after {timeout_minutes} minutes waiting for library installation"
            )
            break

        status: List[LibraryFullStatus] = w.libraries.cluster_status(cluster_id)
        all_finished = True

        for lib in status:
            if lib.status not in final_states:
                all_finished = False
                break

        if all_finished:
            break

        print("Installation in progress, waiting 15 seconds...")
        time.sleep(15)  # Check every 15 seconds

    # Print final status
    status = w.libraries.cluster_status(cluster_id)
    for lib in status:
        if lib.library.pypi:
            status_msg = (
                f"Package: {lib.library.pypi.package} - Status: {lib.status.value}"
            )
            if lib.messages:
                status_msg += f" - Messages: {', '.join(lib.messages)}"
            print(status_msg)


def install_requirements(cluster_id: str, requirements_path: str):
    """Install all packages from requirements.txt into specified cluster."""
    # Initialize workspace client
    w = WorkspaceClient()

    # Parse requirements file
    packages = parse_requirements(requirements_path)

    # Get current library status
    current_status = w.libraries.cluster_status(cluster_id)
    existing_packages = {
        lib.library.pypi.package: lib.status.value
        for lib in current_status
        if lib.library.pypi
    }

    # Filter out already installed packages
    libraries = []
    for package in packages:
        if (
            package not in existing_packages
            or existing_packages[package] != LibraryInstallStatus.INSTALLED.value
        ):
            libraries.append(Library(pypi=PythonPyPiLibrary(package=package)))
        else:
            print(f"Package {package} is already installed, skipping...")

    if not libraries:
        print("All packages are already installed.")
        return

    # Install libraries
    package_names = [lib.pypi.package for lib in libraries]
    print(f"Installing {len(libraries)} packages: {', '.join(package_names)}")
    w.libraries.install(cluster_id, libraries=libraries)

    # Wait for installation to complete
    print("Waiting for installation to complete...")
    wait_for_library_installation(w, cluster_id)
