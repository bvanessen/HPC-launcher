from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    # If type-checking, import the other class
    from hpc_launcher.systems import System

@dataclass
class Scheduler:
    """
    An instance of a batch job scheduler that can launch jobs on a given
    system. Produces command line arguments and scripts to support job
    launching, and provides functionality to interactively or asynchronously
    launch a job.
    """

    # Number of nodes to use
    nodes: int
    # Processes per node
    procs_per_node: int
    # Job name
    job_name: Optional[str] = None
    # Working directory (by default, uses current working directory)
    work_dir: Optional[str] = None
    # Time limit (in minutes), default is no limit
    time_limit: Optional[int] = None
    # The partition or queue to use with the scheduler
    partition: Optional[str] = None
    # The account to use for the scheduler
    account: Optional[str] = None


    def launch_command(self, blocking: bool = True) -> list[str]:
        """
        Returns the launch command for this scheduler. Returns the
        command prefix before the program to run.

        :param blocking: Whether to launch a command that waits for the
                         command to finish (True), or launch a batch
                         script that immediately returns (False).
        :return: The command prefix as a list of strings (one per argument).
        """
        raise NotImplementedError

    def launcher_script(self, system: 'System') -> str:
        """
        Returns the full launcher script, which can be saved as a batch
        script, for the given system and launcher configuration.
        This script usually performs node/resource allocation and manages I/O.

        :param system: The system to use.
        :return: A shell script as a string.
        """
        raise NotImplementedError

    def internal_script(self, system: 'System') -> Optional[str]:
        """
        Returns the script that runs on each process within the allocated job.
        This script is optional, and usually sets up additional elements (e.g.,
        environment variables, working directory, profiler) in case external
        variables are cleaned up by the job scheduler.

        :param system: The system to use.
        :return: A shell script as a string, or None if no internal script is
                 required.
        """
        # By default, no internal script is required
        return None

    def launch(self, system: 'System', program: str,
               args: Optional[list[str]] = None,
               blocking: bool = True,
               verbose: bool = False) -> str:
        """
        Launches the given program and arguments uaing this launcher.

        :param system: The system to use for launching the job.
        :param program: The command line to run.
        :param args: The arguments to use for the program.
        :param blocking: If True, blocks until the job is complete
                         and redirects/duplicates outputs to the terminal.
        :param verbose: If True, prints more information about the job details.
                        This information is also stored in the output log.
        :return: The queued job ID as a string.
        """
        raise NotImplementedError
