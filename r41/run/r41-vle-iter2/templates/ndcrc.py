
import flow
from .ndcrc_scheduler import SGEScheduler

class NotreDameCRC(flow.environment.StandardEnvironment):

    hostname_pattern = r'.*\.crc\.nd\.edu$'
    template = 'crc.nd.sh'
    scheduler_type = SGEScheduler
    JOB_ID_SEPARATOR = '-'


    @classmethod
    def _get_mpi_prefix(cls, operation, parallel):
        """Get the MPI prefix based on the ``nranks`` directives.

        Parameters
        ----------
        operation : :class:`flow.project._JobOperation`
            The operation to be prefixed.
        parallel : bool
            If True, operations are assumed to be executed in parallel, which
            means that the number of total tasks is the sum of all tasks
            instead of the maximum number of tasks. Default is set to False.

        Returns
        -------
        str
            The prefix to be added to the operation's command.

        """
        return ""


#class NotreDameCRCTest(flow.environment.TestEnvironment):
#
#    hostname_pattern = r'.*\.crc\.nd\.edu$'
#    template = 'crc.nd.sh'

