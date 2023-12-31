# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from proto import communication_pb2 as communication__pb2


class TrainingStub(object):
    """服务定义-SL训练
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.SL = channel.unary_unary(
                '/SLFL.Training/SL',
                request_serializer=communication__pb2.SL_Client.SerializeToString,
                response_deserializer=communication__pb2.SL_Server.FromString,
                )


class TrainingServicer(object):
    """服务定义-SL训练
    """

    def SL(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_TrainingServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'SL': grpc.unary_unary_rpc_method_handler(
                    servicer.SL,
                    request_deserializer=communication__pb2.SL_Client.FromString,
                    response_serializer=communication__pb2.SL_Server.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'SLFL.Training', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Training(object):
    """服务定义-SL训练
    """

    @staticmethod
    def SL(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/SLFL.Training/SL',
            communication__pb2.SL_Client.SerializeToString,
            communication__pb2.SL_Server.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class ValidationStub(object):
    """服务定义-SL验证
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.val = channel.unary_unary(
                '/SLFL.Validation/val',
                request_serializer=communication__pb2.SL_Client_Eval.SerializeToString,
                response_deserializer=communication__pb2.ValidationResponse.FromString,
                )


class ValidationServicer(object):
    """服务定义-SL验证
    """

    def val(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ValidationServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'val': grpc.unary_unary_rpc_method_handler(
                    servicer.val,
                    request_deserializer=communication__pb2.SL_Client_Eval.FromString,
                    response_serializer=communication__pb2.ValidationResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'SLFL.Validation', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Validation(object):
    """服务定义-SL验证
    """

    @staticmethod
    def val(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/SLFL.Validation/val',
            communication__pb2.SL_Client_Eval.SerializeToString,
            communication__pb2.ValidationResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
