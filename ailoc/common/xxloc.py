from abc import ABC, abstractmethod  # abstract class


class XXLoc(ABC):
    """
    Abstract class, all XXLoc should inherit from this class.
    """

    @property
    @abstractmethod
    def network(self, *args, **kwargs):
        """
        XXLoc should have a network instance as a property.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def data_simulator(self, *args, **kwargs):
        """
        XXLoc must have a data simulator instance as a property.
        """

        raise NotImplementedError

    @abstractmethod
    def online_train(self, *args, **kwargs):
        """
        Train the network with training data generated online. Maybe the most important method of XXLoc.
        """

        raise NotImplementedError

    @abstractmethod
    def inference(self, *args, **kwargs):
        """
        Inference with the network.
        """

        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, *args, **kwargs):
        """
        Loss function.
        """

        raise NotImplementedError

    @abstractmethod
    def post_process(self, *args, **kwargs):
        """
        Post process the inference output, should return the molecule list.
        """

        raise NotImplementedError

    @abstractmethod
    def analyze(self, *args, **kwargs):
        """
        Wrapper function used for experimental data analysis, receive a batch of data and
        return the molecule list, should contain the pre-process, inference and post-process procedures.
        """

        raise NotImplementedError

    @abstractmethod
    def online_evaluate(self, *args, **kwargs):
        """
        Evaluate the network using the validation dataset generated online.
        """

        raise NotImplementedError

    @abstractmethod
    def save(self, *args, **kwargs):
        """
        Save the network.
        """

        raise NotImplementedError

