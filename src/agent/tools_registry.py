"""
Tools Registry for external APIs, calculators, and utility functions.

This module provides a centralized registry of callable tools that the agent
can use to fetch external data or perform computations.

Design principles:
- Each tool is a callable function with clear input/output contracts
- Tools are self-documenting with metadata
- Easy to extend with new tools
- Supports both sync and async tools
"""

import logging
from typing import Any, Dict, List, Callable, Optional
from datetime import datetime
import json

logger = logging.getLogger("uvicorn.error")


class Tool:
    """
    Represents a single callable tool.
    """
    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: Dict[str, Any],
        is_async: bool = False,
    ):
        self.name = name
        self.description = description
        self.function = function
        self.parameters = parameters  # {"param_name": {"type": "str", "required": True, "description": "..."}}
        self.is_async = is_async

    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        try:
            if self.is_async:
                return await self.function(**kwargs)
            else:
                return self.function(**kwargs)
        except Exception as exc:
            logger.error(f"[Tool:{self.name}] Execution failed: {exc}")
            return {"error": str(exc)}

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool metadata to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "is_async": self.is_async,
        }


class ToolsRegistry:
    """
    Central registry for all available tools.
    """
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register built-in tools."""
        
        # ── Calculator Tool ──────────────────────────────────
        def calculator(expression: str) -> Dict[str, Any]:
            """
            Evaluate a mathematical expression.
            Supports basic arithmetic: +, -, *, /, **, (), etc.
            """
            try:
                # Security: only allow safe mathematical operations
                allowed_chars = set("0123456789+-*/().** ")
                if not all(c in allowed_chars for c in expression.replace(" ", "")):
                    return {"error": "Invalid characters in expression"}
                
                result = eval(expression, {"__builtins__": {}}, {})
                return {
                    "result": result,
                    "expression": expression,
                    "success": True,
                }
            except Exception as exc:
                return {"error": str(exc), "success": False}

        self.register_tool(
            name="calculator",
            description="Evaluate mathematical expressions. Supports +, -, *, /, **, parentheses.",
            function=calculator,
            parameters={
                "expression": {
                    "type": "string",
                    "required": True,
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5 + 3')",
                }
            },
        )

        # ── Current Time Tool ────────────────────────────────
        def get_current_time(timezone: str = "UTC") -> Dict[str, Any]:
            """
            Get current date and time.
            """
            try:
                now = datetime.now()
                return {
                    "datetime": now.isoformat(),
                    "date": now.strftime("%Y-%m-%d"),
                    "time": now.strftime("%H:%M:%S"),
                    "timezone": timezone,
                    "timestamp": now.timestamp(),
                    "success": True,
                }
            except Exception as exc:
                return {"error": str(exc), "success": False}

        self.register_tool(
            name="get_current_time",
            description="Get current date and time information.",
            function=get_current_time,
            parameters={
                "timezone": {
                    "type": "string",
                    "required": False,
                    "description": "Timezone (default: UTC)",
                    "default": "UTC",
                }
            },
        )

        # ── Unit Converter Tool ──────────────────────────────
        def unit_converter(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
            """
            Convert between common units.
            Supports: length, weight, temperature
            """
            try:
                conversions = {
                    # Length (to meters)
                    "m": 1.0,
                    "km": 1000.0,
                    "cm": 0.01,
                    "mm": 0.001,
                    "mile": 1609.34,
                    "yard": 0.9144,
                    "foot": 0.3048,
                    "inch": 0.0254,
                    
                    # Weight (to kg)
                    "kg": 1.0,
                    "g": 0.001,
                    "mg": 0.000001,
                    "lb": 0.453592,
                    "oz": 0.0283495,
                    
                    # Temperature (special handling)
                }
                
                # Temperature conversion (special case)
                if from_unit.lower() in ["c", "f", "k"] or to_unit.lower() in ["c", "f", "k"]:
                    # Celsius conversions
                    if from_unit.lower() == "c":
                        if to_unit.lower() == "f":
                            result = (value * 9/5) + 32
                        elif to_unit.lower() == "k":
                            result = value + 273.15
                        else:
                            result = value
                    # Fahrenheit conversions
                    elif from_unit.lower() == "f":
                        if to_unit.lower() == "c":
                            result = (value - 32) * 5/9
                        elif to_unit.lower() == "k":
                            result = (value - 32) * 5/9 + 273.15
                        else:
                            result = value
                    # Kelvin conversions
                    elif from_unit.lower() == "k":
                        if to_unit.lower() == "c":
                            result = value - 273.15
                        elif to_unit.lower() == "f":
                            result = (value - 273.15) * 9/5 + 32
                        else:
                            result = value
                    else:
                        return {"error": "Invalid temperature unit", "success": False}
                else:
                    # Standard unit conversion
                    if from_unit not in conversions or to_unit not in conversions:
                        return {"error": f"Unsupported units: {from_unit} or {to_unit}", "success": False}
                    
                    # Convert to base unit, then to target unit
                    base_value = value * conversions[from_unit]
                    result = base_value / conversions[to_unit]
                
                return {
                    "result": result,
                    "from_value": value,
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "success": True,
                }
            except Exception as exc:
                return {"error": str(exc), "success": False}

        self.register_tool(
            name="unit_converter",
            description="Convert between units (length, weight, temperature). Examples: m, km, kg, lb, C, F",
            function=unit_converter,
            parameters={
                "value": {
                    "type": "number",
                    "required": True,
                    "description": "Numeric value to convert",
                },
                "from_unit": {
                    "type": "string",
                    "required": True,
                    "description": "Source unit (e.g., 'km', 'lb', 'C')",
                },
                "to_unit": {
                    "type": "string",
                    "required": True,
                    "description": "Target unit (e.g., 'm', 'kg', 'F')",
                },
            },
        )

        # ── JSON Parser Tool ─────────────────────────────────
        def json_parser(json_string: str) -> Dict[str, Any]:
            """
            Parse and validate JSON strings.
            """
            try:
                parsed = json.loads(json_string)
                return {
                    "parsed": parsed,
                    "valid": True,
                    "success": True,
                }
            except json.JSONDecodeError as exc:
                return {
                    "error": str(exc),
                    "valid": False,
                    "success": False,
                }

        self.register_tool(
            name="json_parser",
            description="Parse and validate JSON strings.",
            function=json_parser,
            parameters={
                "json_string": {
                    "type": "string",
                    "required": True,
                    "description": "JSON string to parse",
                }
            },
        )

    def register_tool(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: Dict[str, Any],
        is_async: bool = False,
    ) -> None:
        """
        Register a new tool in the registry.
        
        Parameters
        ----------
        name : str
            Unique tool identifier
        description : str
            Human-readable description of what the tool does
        function : Callable
            The actual function to execute
        parameters : Dict[str, Any]
            Parameter schema for the tool
        is_async : bool
            Whether the function is async
        """
        tool = Tool(name, description, function, parameters, is_async)
        self.tools[name] = tool
        logger.info(f"[ToolsRegistry] Registered tool: {name}")

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools with their metadata."""
        return [tool.to_dict() for tool in self.tools.values()]

    def get_tool_names(self) -> List[str]:
        """Get list of all tool names."""
        return list(self.tools.keys())

    async def execute_tool(self, name: str, **kwargs) -> Any:
        """
        Execute a tool by name with given parameters.
        
        Parameters
        ----------
        name : str
            Tool name
        **kwargs
            Tool parameters
            
        Returns
        -------
        Tool execution result
        """
        tool = self.get_tool(name)
        if not tool:
            logger.error(f"[ToolsRegistry] Tool not found: {name}")
            return {"error": f"Tool '{name}' not found", "success": False}

        logger.info(f"[ToolsRegistry] Executing tool: {name} with params: {kwargs}")
        result = await tool.execute(**kwargs)
        logger.info(f"[ToolsRegistry] Tool {name} result: {result}")
        return result

    def get_tools_summary(self) -> str:
        """
        Get a human-readable summary of all available tools.
        Useful for prompts.
        """
        if not self.tools:
            return "No tools available"
        
        summary = []
        for tool in self.tools.values():
            params = ", ".join([
                f"{name}({info['type']})"
                for name, info in tool.parameters.items()
            ])
            summary.append(f"- {tool.name}({params}): {tool.description}")
        
        return "\n".join(summary)
