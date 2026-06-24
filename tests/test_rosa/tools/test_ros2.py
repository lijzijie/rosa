#  Copyright (c) 2024. Jet Propulsion Laboratory. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import subprocess
import unittest
from unittest.mock import patch, call

try:
    from src.rosa.tools.ros2 import (
        execute_ros_command,
        _validate_ros_arg,
        ros2_node_list,
        ros2_topic_list,
        ros2_topic_echo,
        ros2_service_list,
        ros2_node_info,
        ros2_param_list,
        ros2_param_get,
        ros2_param_set,
        ros2_service_call,
    )
except ModuleNotFoundError:
    pass


@unittest.skipIf(
    os.environ.get("ROS_VERSION") == "1",
    "Skipping ROS2 tests because ROS_VERSION is set to 1",
)
class TestROS2Tools(unittest.TestCase):

    # ------------------------------------------------------------------ #
    #  execute_ros_command: basic functionality
    # ------------------------------------------------------------------ #

    @patch("src.rosa.tools.ros2.subprocess.check_output")
    def test_execute_valid_ros2_command_string(self, mock_check_output):
        """Backward-compat: accept a plain string and split it."""
        mock_check_output.return_value = b"Node /example_node\n"
        success, output = execute_ros_command("ros2 node list")
        self.assertTrue(success)
        self.assertEqual(output, "Node /example_node\n")
        # The critical assertion: shell=False must be used
        mock_check_output.assert_called_once_with(
            ["ros2", "node", "list"], shell=False
        )

    @patch("src.rosa.tools.ros2.subprocess.check_output")
    def test_execute_valid_ros2_command_list(self, mock_check_output):
        """Preferred path: accept a pre-split list."""
        mock_check_output.return_value = b"/topic1\n/topic2\n"
        success, output = execute_ros_command(["ros2", "topic", "list"])
        self.assertTrue(success)
        self.assertEqual(output, "/topic1\n/topic2\n")
        mock_check_output.assert_called_once_with(
            ["ros2", "topic", "list"], shell=False
        )

    @patch("src.rosa.tools.ros2.subprocess.check_output")
    def test_execute_invalid_ros2_command(self, mock_check_output):
        mock_check_output.side_effect = subprocess.CalledProcessError(
            1, "ros2 node list"
        )
        success, output = execute_ros_command("ros2 node list")
        self.assertFalse(success)
        self.assertIn(
            "Command 'ros2 node list' returned non-zero exit status 1.", output
        )

    def test_execute_command_with_invalid_prefix(self):
        with self.assertRaises(ValueError):
            execute_ros_command("invalid node list")

    def test_execute_command_with_invalid_subcommand(self):
        with self.assertRaises(ValueError):
            execute_ros_command("ros2 invalid_subcommand")

    def test_execute_command_with_insufficient_arguments(self):
        with self.assertRaises(ValueError):
            execute_ros_command("ros2")

    # ------------------------------------------------------------------ #
    #  shell=False enforcement (the core security assertion)
    # ------------------------------------------------------------------ #

    @patch("src.rosa.tools.ros2.subprocess.check_output")
    def test_shell_false_is_always_used(self, mock_check_output):
        """Ensure every call goes through shell=False."""
        mock_check_output.return_value = b"ok\n"
        execute_ros_command("ros2 node list")
        for c in mock_check_output.call_args_list:
            self.assertEqual(c.kwargs.get("shell", c[1].get("shell")), False)

    # ------------------------------------------------------------------ #
    #  Input sanitisation: _validate_ros_arg
    # ------------------------------------------------------------------ #

    def test_validate_ros_arg_accepts_normal_names(self):
        """Normal ROS names should pass validation."""
        for name in ["/turtle1/cmd_vel", "/node_1", "std_srvs/srv/Empty",
                     "my_param", "0.5", "/ns/sub/topic"]:
            _validate_ros_arg(name)  # should not raise

    def test_validate_ros_arg_rejects_semicolon(self):
        with self.assertRaises(ValueError):
            _validate_ros_arg("/topic; rm -rf /")

    def test_validate_ros_arg_rejects_pipe(self):
        with self.assertRaises(ValueError):
            _validate_ros_arg("/topic | cat /etc/passwd")

    def test_validate_ros_arg_rejects_backtick(self):
        with self.assertRaises(ValueError):
            _validate_ros_arg("`whoami`")

    def test_validate_ros_arg_rejects_dollar(self):
        with self.assertRaises(ValueError):
            _validate_ros_arg("$(cat /etc/shadow)")

    def test_validate_ros_arg_rejects_ampersand(self):
        with self.assertRaises(ValueError):
            _validate_ros_arg("/topic & malicious_cmd")

    def test_validate_ros_arg_rejects_newline(self):
        with self.assertRaises(ValueError):
            _validate_ros_arg("/topic\nmalicious_cmd")

    # ------------------------------------------------------------------ #
    #  Shell injection attack scenarios
    # ------------------------------------------------------------------ #

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_topic_echo_rejects_injection_in_topic(self, mock_execute):
        """LLM-generated topic name with injection payload must be rejected."""
        result = ros2_topic_echo.invoke(
            {"topic": "/topic; rm -rf /", "count": 1}
        )
        # The tool should never reach execute_ros_command
        mock_execute.assert_not_called()

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_node_info_rejects_injection(self, mock_execute):
        """Node name with injection payload must be rejected."""
        result = ros2_node_info.invoke({"nodes": ["/node`whoami`"]})
        mock_execute.assert_not_called()
        self.assertIn("error", result.get("/node`whoami`", {}))

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_param_set_rejects_injection_in_value(self, mock_execute):
        """Parameter value with injection payload must be rejected."""
        result = ros2_param_set.invoke({
            "node_name": "/example_node",
            "param_name": "safe_param",
            "param_value": "value; cat /etc/passwd",
        })
        mock_execute.assert_not_called()

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_service_call_rejects_injection_in_service_name(self, mock_execute):
        """Service name with injection payload must be rejected."""
        result = ros2_service_call.invoke({
            "service_name": "/srv$(whoami)",
            "srv_type": "std_srvs/srv/Empty",
            "request": "{}",
        })
        mock_execute.assert_not_called()

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_service_call_rejects_injection_in_type(self, mock_execute):
        """Service type with injection payload must be rejected."""
        result = ros2_service_call.invoke({
            "service_name": "/safe_service",
            "srv_type": "std_srvs; rm -rf /",
            "request": "{}",
        })
        mock_execute.assert_not_called()

    # ------------------------------------------------------------------ #
    #  Existing functional tests (updated for list-based command passing)
    # ------------------------------------------------------------------ #

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_node_list_returns_nodes(self, mock_execute):
        mock_execute.return_value = (True, "/node1\n/node2\n")
        result = ros2_node_list.invoke({"pattern": None})
        self.assertEqual(result, {"nodes": ["/node1", "/node2"]})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_node_list_with_pattern(self, mock_execute):
        mock_execute.return_value = (True, "/node1\n/node2\n")
        result = ros2_node_list.invoke({"pattern": "node1"})
        self.assertEqual(result, {"nodes": ["/node1"]})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_node_list_with_blacklist(self, mock_execute):
        mock_execute.return_value = (True, "/node1\n/node2\n")
        result = ros2_node_list.invoke({"blacklist": ["node2"]})
        self.assertEqual(result, {"nodes": ["/node1"]})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_node_list_invalid_command(self, mock_execute):
        mock_execute.return_value = (False, "Invalid command")
        result = ros2_node_list.invoke({"pattern": None})
        self.assertEqual(result, {"nodes": ["Invalid command"]})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_topic_list_returns_topics(self, mock_execute):
        mock_execute.return_value = (True, "/topic1\n/topic2\n")
        result = ros2_topic_list.invoke({"pattern": None})
        self.assertEqual(result, {"topics": ["/topic1", "/topic2"]})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_topic_list_with_pattern(self, mock_execute):
        mock_execute.return_value = (True, "/topic1\n/topic2\n")
        result = ros2_topic_list.invoke({"pattern": "topic1"})
        self.assertEqual(result, {"topics": ["/topic1"]})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_topic_list_with_blacklist(self, mock_execute):
        mock_execute.return_value = (True, "/topic1\n/topic2\n")
        result = ros2_topic_list.invoke({"blacklist": ["topic2"]})
        self.assertEqual(result, {"topics": ["/topic1"]})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_topic_list_invalid_command(self, mock_execute):
        mock_execute.return_value = (False, "Invalid command")
        result = ros2_topic_list.invoke({"pattern": None})
        self.assertEqual(result, {"topics": ["Invalid command"]})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_topic_echo_success(self, mock_execute):
        mock_execute.return_value = (True, "Message 1\n")
        result = ros2_topic_echo.invoke(
            {"topic": "/example_topic", "count": 1, "return_echoes": True}
        )
        self.assertEqual(result, {"echoes": ["Message 1\n"]})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_topic_echo_multiple_messages(self, mock_execute):
        mock_execute.return_value = (True, "Message 1\n")
        result = ros2_topic_echo.invoke(
            {"topic": "/example_topic", "count": 3, "return_echoes": True}
        )
        self.assertEqual(
            result, {"echoes": ["Message 1\n", "Message 1\n", "Message 1\n"]}
        )

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_topic_echo_invalid_topic(self, mock_execute):
        mock_execute.return_value = (False, "Invalid topic")
        result = ros2_topic_echo.invoke({"topic": "/invalid_topic", "count": 1})
        self.assertEqual(result, {"error": "Invalid topic"})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_topic_echo_invalid_count(self, mock_execute):
        result = ros2_topic_echo.invoke({"topic": "/example_topic", "count": 11})
        self.assertEqual(result, {"error": "Count must be between 1 and 10."})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_topic_echo_command_failure(self, mock_execute):
        mock_execute.return_value = (False, "Command failed")
        result = ros2_topic_echo.invoke({"topic": "/example_topic", "count": 1})
        self.assertEqual(result, {"error": "Command failed"})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_service_list_returns_services(self, mock_execute):
        mock_execute.return_value = (True, "/service1\n/service2\n")
        result = ros2_service_list.invoke({"pattern": None})
        self.assertEqual(result, {"services": ["/service1", "/service2"]})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_service_list_with_pattern(self, mock_execute):
        mock_execute.return_value = (True, "/service1\n/service2\n")
        result = ros2_service_list.invoke({"pattern": "service1"})
        self.assertEqual(result, {"services": ["/service1"]})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_service_list_with_blacklist(self, mock_execute):
        mock_execute.return_value = (True, "/service1\n/service2\n")
        result = ros2_service_list.invoke({"blacklist": ["service2"]})
        self.assertEqual(result, {"services": ["/service1"]})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_service_list_invalid_command(self, mock_execute):
        mock_execute.return_value = (False, "Invalid command")
        result = ros2_service_list.invoke({"pattern": None})
        self.assertEqual(result, {"services": ["Invalid command"]})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_node_info_success(self, mock_execute):
        mock_execute.return_value = (True, "Node info for /node1")
        result = ros2_node_info.invoke({"nodes": ["/node1"]})
        self.assertEqual(result, {"/node1": "Node info for /node1"})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_node_info_multiple_nodes(self, mock_execute):
        mock_execute.side_effect = [
            (True, "Node info for /node1"),
            (True, "Node info for /node2"),
        ]
        result = ros2_node_info.invoke({"nodes": ["/node1", "/node2"]})
        self.assertEqual(
            result, {"/node1": "Node info for /node1", "/node2": "Node info for /node2"}
        )

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_node_info_invalid_node(self, mock_execute):
        mock_execute.return_value = (False, "Invalid node")
        result = ros2_node_info.invoke({"nodes": ["/invalid_node"]})
        self.assertEqual(result, {"/invalid_node": {"error": "Invalid node"}})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_node_info_command_failure(self, mock_execute):
        mock_execute.return_value = (False, "Command failed")
        result = ros2_node_info.invoke({"nodes": ["/node1"]})
        self.assertEqual(result, {"/node1": {"error": "Command failed"}})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_param_list_returns_params_for_node(self, mock_execute):
        mock_execute.return_value = (True, "param1\nparam2\n")
        result = ros2_param_list.invoke({"node_name": "/example_node"})
        self.assertEqual(result, {"/example_node": ["param1", "param2"]})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_param_list_returns_all_params(self, mock_execute):
        mock_execute.return_value = (
            True,
            "/node1\n  param1\n  param2\n/node2\n  param3\n",
        )
        result = ros2_param_list.invoke({})
        self.assertEqual(result, {"/node1": ["param1", "param2"], "/node2": ["param3"]})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_param_list_with_pattern(self, mock_execute):
        mock_execute.return_value = (True, "param1\nparam2\n")
        result = ros2_param_list.invoke(
            {"node_name": "/example_node", "pattern": "param1"}
        )
        self.assertEqual(result, {"/example_node": ["param1"]})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_param_list_with_blacklist(self, mock_execute):
        mock_execute.return_value = (True, "param1\nparam2\n")
        result = ros2_param_list.invoke(
            {"node_name": "/example_node", "blacklist": ["param2"]}
        )
        self.assertEqual(result, {"/example_node": ["param1"]})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_param_list_invalid_command(self, mock_execute):
        mock_execute.return_value = (False, "Invalid command")
        result = ros2_param_list.invoke({"node_name": "/example_node"})
        self.assertEqual(result, {"error": "Invalid command"})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_param_get_success(self, mock_execute):
        mock_execute.return_value = (True, "value1")
        result = ros2_param_get.invoke(
            {"node_name": "/example_node", "param_name": "param1"}
        )
        self.assertEqual(result, {"param1": "value1"})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_param_get_invalid_command(self, mock_execute):
        mock_execute.return_value = (False, "Invalid command")
        result = ros2_param_get.invoke(
            {"node_name": "/example_node", "param_name": "param1"}
        )
        self.assertEqual(result, {"error": "Invalid command"})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_param_set_success(self, mock_execute):
        mock_execute.return_value = (True, "value1")
        result = ros2_param_set.invoke(
            {
                "node_name": "/example_node",
                "param_name": "param1",
                "param_value": "value1",
            }
        )
        self.assertEqual(result, {"param1": "value1"})

    @patch("src.rosa.tools.ros2.execute_ros_command")
    def test_ros2_param_set_invalid_command(self, mock_execute):
        mock_execute.return_value = (False, "Invalid command")
        result = ros2_param_set.invoke(
            {
                "node_name": "/example_node",
                "param_name": "param1",
                "param_value": "value1",
            }
        )
        self.assertEqual(result, {"error": "Invalid command"})


if __name__ == "__main__":
    import os

    if os.environ.get("ROS_VERSION") == 2:
        unittest.main()
