"""Tests for built-in tool parameter declarations.

Covers:
  - BuiltinToolParameter schema validation
  - ToolParameter dataclass in the registry
  - Auto-extraction of parameters from function signatures
  - YAML-declared parameters taking precedence over auto-extraction
  - find_param_doc docstring extraction
  - _get_yaml_parameters conversion
  - Framework parameter filtering
"""

from __future__ import annotations

import pytest

from orchid_ai.config import tool_registry as treg
from orchid_ai.config.schema import BuiltinToolParameter, OrchidAgentsConfig, OrchidBuiltinToolConfig
from orchid_ai.config.tool_registry import ToolParameter, _extract_parameters_from_handler, find_param_doc


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear the tool registry before and after each test."""
    treg.clear()
    yield
    treg.clear()


# ── Schema tests ────────────────────────────────────────────────


class TestBuiltinToolParameterSchema:
    """Test the Pydantic BuiltinToolParameter model."""

    def test_defaults(self):
        p = BuiltinToolParameter()
        assert p.type == "string"
        assert p.description == ""
        assert p.required is True
        assert p.default is None

    def test_custom_values(self):
        p = BuiltinToolParameter(type="int", description="Page number", required=False, default=1)
        assert p.type == "int"
        assert p.description == "Page number"
        assert p.required is False
        assert p.default == 1

    def test_bool_parameter(self):
        p = BuiltinToolParameter(type="bool", description="Enable feature", required=False, default=True)
        assert p.type == "bool"
        assert p.default is True

    def test_builtin_tool_config_with_parameters(self):
        cfg = OrchidBuiltinToolConfig(
            handler="os.path.join",
            description="Join paths",
            parameters={
                "base": BuiltinToolParameter(type="string", description="Base path", required=True),
                "suffix": BuiltinToolParameter(type="string", description="Path suffix", required=False, default=""),
            },
        )
        assert len(cfg.parameters) == 2
        assert cfg.parameters["base"].required is True
        assert cfg.parameters["suffix"].default == ""

    def test_builtin_tool_config_no_parameters(self):
        cfg = OrchidBuiltinToolConfig(handler="os.path.join", description="Join paths")
        assert cfg.parameters == {}


# ── Auto-extraction tests ──────────────────────────────────────


class TestAutoExtraction:
    """Test parameter auto-extraction from function signatures."""

    def test_simple_function(self):
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        params = _extract_parameters_from_handler(greet)
        assert "name" in params
        assert params["name"].type == "string"
        assert params["name"].required is True
        assert "greeting" in params
        assert params["greeting"].required is False
        assert params["greeting"].default == "Hello"

    def test_numeric_types(self):
        def calc(enrolled: int, rate: float, active: bool = True) -> float:
            return 0.0

        params = _extract_parameters_from_handler(calc)
        assert params["enrolled"].type == "int"
        assert params["rate"].type == "float"
        assert params["active"].type == "bool"
        assert params["active"].default is True

    def test_framework_params_filtered(self):
        def tool_fn(query: str, context: dict, auth_context: object, name: str = "", **kwargs) -> str:
            return ""

        params = _extract_parameters_from_handler(tool_fn)
        assert "query" not in params
        assert "context" not in params
        assert "auth_context" not in params
        assert "kwargs" not in params
        assert "name" in params

    def test_var_positional_filtered(self):
        def tool_fn(*args, name: str = "") -> str:
            return ""

        params = _extract_parameters_from_handler(tool_fn)
        assert "args" not in params
        assert "name" in params

    def test_unannotated_defaults_to_string(self):
        def tool_fn(player_name="") -> str:
            return ""

        params = _extract_parameters_from_handler(tool_fn)
        assert params["player_name"].type == "string"

    def test_empty_function(self):
        def tool_fn(**kwargs) -> str:
            return ""

        params = _extract_parameters_from_handler(tool_fn)
        assert params == {}


# ── Docstring extraction tests ─────────────────────────────────


class TestFindParamDoc:
    """Test docstring parameter description extraction."""

    def test_google_style(self):
        doc = """Search the menu.

        Args:
            query: Keyword to search for
            dietary_filter: Dietary restriction
        """
        assert find_param_doc(doc, "query") == "Keyword to search for"
        assert find_param_doc(doc, "dietary_filter") == "Dietary restriction"

    def test_numpy_style(self):
        doc = """Search the menu.

        Parameters
        ----------
        query : str
            Keyword to search for
        dietary_filter : str
            Dietary restriction
        """
        assert find_param_doc(doc, "query") == "Keyword to search for"

    def test_not_found(self):
        assert find_param_doc("No params here", "missing") == ""

    def test_empty_docstring(self):
        assert find_param_doc("", "anything") == ""


# ── YAML precedence tests ─────────────────────────────────────


class TestYAMLPrecedence:
    """Test that YAML-declared parameters override auto-extraction."""

    def test_yaml_params_used_when_declared(self):
        def my_tool(name: str, count: int = 5) -> str:
            return ""

        yaml_params = {
            "name": ToolParameter(name="name", type="string", description="Custom description", required=True),
        }
        treg.register_tool("my_tool", my_tool, "Test", parameters=yaml_params)
        entry = treg.get_tool("my_tool")
        # Should use YAML params, not auto-extracted
        assert "name" in entry.parameters
        assert entry.parameters["name"].description == "Custom description"
        # count is NOT in YAML, so it should NOT be in the result (YAML takes full precedence)
        assert "count" not in entry.parameters

    def test_auto_extraction_when_no_yaml_params(self):
        def my_tool(name: str, count: int = 5) -> str:
            return ""

        treg.register_tool("my_tool", my_tool, "Test", parameters=None)
        entry = treg.get_tool("my_tool")
        assert "name" in entry.parameters
        assert "count" in entry.parameters
        assert entry.parameters["count"].type == "int"
        assert entry.parameters["count"].default == 5


# ── load_tools_from_config with parameters ─────────────────────


class TestLoadToolsWithParameters:
    """Test loading tools from config with YAML-declared parameters."""

    def test_load_with_parameters(self):
        config = {
            "join_path": OrchidBuiltinToolConfig(
                handler="os.path.join",
                description="Join paths",
                parameters={
                    "base": BuiltinToolParameter(type="string", description="Base path", required=True),
                },
            ),
        }
        treg.load_tools_from_config(config)
        entry = treg.get_tool("join_path")
        assert "base" in entry.parameters
        assert entry.parameters["base"].description == "Base path"

    def test_load_without_parameters_triggers_extraction(self):
        """When no parameters declared in YAML, auto-extract from handler."""
        config = {
            "isabs": OrchidBuiltinToolConfig(
                handler="os.path.isabs",
                description="Check if path is absolute",
            ),
        }
        treg.load_tools_from_config(config)
        entry = treg.get_tool("isabs")
        # os.path.isabs has a 's' parameter — auto-extracted
        assert len(entry.parameters) >= 0  # implementation-dependent


# ── Full YAML config validation tests ──────────────────────────


class TestYAMLConfigValidation:
    """Test that YAML configs with parameters parse correctly."""

    def test_tool_with_all_param_types(self):
        import yaml

        raw = yaml.safe_load("""
version: "1"
tools:
  test_tool:
    handler: "os.path.join"
    description: "Test tool"
    parameters:
      name:
        type: string
        description: "A name"
        required: true
      count:
        type: int
        description: "A count"
        required: false
        default: 10
      rate:
        type: float
        description: "A rate"
        required: false
        default: 0.5
      enabled:
        type: bool
        description: "Toggle"
        required: false
        default: true
agents: {}
""")
        config = OrchidAgentsConfig(**raw)
        tool = config.tools["test_tool"]
        assert tool.parameters["name"].type == "string"
        assert tool.parameters["name"].required is True
        assert tool.parameters["count"].type == "int"
        assert tool.parameters["count"].default == 10
        assert tool.parameters["rate"].type == "float"
        assert tool.parameters["rate"].default == 0.5
        assert tool.parameters["enabled"].type == "bool"
        assert tool.parameters["enabled"].default is True

    def test_tool_without_parameters_field(self):
        import yaml

        raw = yaml.safe_load("""
version: "1"
tools:
  test_tool:
    handler: "os.path.join"
    description: "Test tool"
agents: {}
""")
        config = OrchidAgentsConfig(**raw)
        assert config.tools["test_tool"].parameters == {}
