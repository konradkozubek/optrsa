import inspect
import argparse
from typing import Optional, List, Callable


class ModuleArgumentParser:
    def __init__(self) -> None:
        # Get calling module docstring to use it as argument parser description
        # TODO Check if it works in Python 3.10
        # from __main__ import __doc__ as module_docstring
        # description = inspect.cleandoc(module_docstring)
        calling_module = inspect.getmodule(inspect.currentframe().f_back)
        description = inspect.cleandoc(calling_module.__doc__)

        # TODO Maybe use configargparse instead of argparse
        self.arg_parser = argparse.ArgumentParser(description=description,
                                                  formatter_class=argparse.RawTextHelpFormatter)
        self.arg_subparsers = self.arg_parser.add_subparsers(title="program modes",
                                                             description="available subcommands",
                                                             dest="subcommand")
        self.arg_subparsers.required = True
        # The above statement and the dest argument to add_subparsers method is added in order to make subcommand
        # required.
        # Since Python 3.7 required=True argument of add_subparsers method could be used instead.
        # See https://docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.add_subparsers and
        # https://stackoverflow.com/questions/23349349/argparse-with-required-subparser
        self.common_arg_parsers = {}

    def add_common_arg_parser(self, name: str) -> argparse.ArgumentParser:
        common_arg_parser = argparse.ArgumentParser(add_help=False)
        self.common_arg_parsers[name] = common_arg_parser
        return common_arg_parser

    @staticmethod
    def get_help_from_doc(doc: str) -> str:
        help_text = doc.split("\n")[0]
        if help_text[-1] == ".":
            help_text = help_text[:-1]
        return help_text

    # TODO Try to annotate return type of this function
    def command(self, name: Optional[str] = None, parsers: Optional[List[str]] = None):
        def command_decorator(function: Callable[..., None]) -> Callable[..., None]:
            # See https://stackoverflow.com/questions/2609518/unboundlocalerror-with-nested-function-scopes
            nonlocal name
            if name is None:
                name = function.__name__
            if parsers is not None:
                parent_arg_parsers = [self.common_arg_parsers[parser_name] for parser_name in parsers]
            else:
                parent_arg_parsers = []
            command_arg_subparser = self.arg_subparsers.add_parser(name,
                                                                   parents=parent_arg_parsers,
                                                                   description=inspect.getdoc(function)
                                                                   .partition("\n:")[0],
                                                                   formatter_class=argparse.RawTextHelpFormatter,
                                                                   help=self.get_help_from_doc(
                                                                       inspect.getdoc(function)))
            command_arg_subparser.set_defaults(mode=function)
            function.command_arg_subparser = command_arg_subparser
            return function
        return command_decorator

    @staticmethod
    def argument(*args, **kwargs):
        def argument_decorator(function: Callable[..., None]) -> Callable[..., None]:
            if hasattr(function, "command_arg_subparser"):
                function.command_arg_subparser.add_argument(*args, **kwargs)
            else:
                raise TypeError("Decorated function is not a command")
            return function
        return argument_decorator

    def __call__(self) -> None:
        args = self.arg_parser.parse_args()
        args_dict = vars(args)
        del args_dict["subcommand"]
        mode = args_dict.pop("mode")
        mode(**args_dict)
