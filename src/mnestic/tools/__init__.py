"""Typed tool abstraction with workspace confinement."""

from mnestic.tools.base import Tool, ToolContext, ToolError, ToolRegistry, ToolResult
from mnestic.tools.filesystem import ListDirectoryTool, ReadTextFileTool, SearchTextTool, WriteWorkspaceFileTool
from mnestic.tools.memory_tool import ArchivalMemorySearchTool
from mnestic.tools.shell import ShellTool


def default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for tool in (
        ReadTextFileTool(), ListDirectoryTool(), SearchTextTool(), WriteWorkspaceFileTool(), ShellTool(),
        ArchivalMemorySearchTool(),
    ):
        reg.register(tool)
    return reg


__all__ = [
    "ArchivalMemorySearchTool", "ListDirectoryTool", "ReadTextFileTool", "SearchTextTool", "ShellTool", "Tool",
    "ToolContext", "ToolError", "ToolRegistry", "ToolResult", "WriteWorkspaceFileTool", "default_registry",
]
