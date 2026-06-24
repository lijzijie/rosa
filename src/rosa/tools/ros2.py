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
import re
import shlex
import subprocess
import time
from typing import List, Optional, Tuple

from langchain.agents import tool
from rclpy.logging import get_logging_directory

# Detection and configuration for native ROS2 rclpy support
HAS_RCLPY = False
try:
    import rclpy
    from rclpy.node import Node
    HAS_RCLPY = True
except ImportError:
    pass

class ROSANode(Node if HAS_RCLPY else object):
    """Singleton ROS2 node for high-performance native tool operations."""
    _instance = None

    @classmethod
    def get_instance(cls):
        if not HAS_RCLPY:
            return None
        if cls._instance is None:
            try:
                if not rclpy.ok():
                    rclpy.init()
                cls._instance = cls("rosa_agent_node")
            except Exception:
                cls._instance = None
        return cls._instance


# Security validation: reject shell metacharacters in dynamic inputs
_SHELL_META_CHARS = re.compile(r"[;&|`$(){}!\n\r\\]")

def _validate_ros_arg(value: str, label: str = "argument") -> None:
    if _SHELL_META_CHARS.search(value):
        raise ValueError(
            f"Invalid {label}: '{value}' contains disallowed shell metacharacters."
        )


def execute_ros_command(command) -> Tuple[bool, str]:
    """
    Execute a ROS2 command.
    Accepts either a pre-split list of arguments (preferred) or a single
    command string for backward compatibility. In both cases the command
    is executed without a shell (shell=False) to prevent command injection.
    """
    if isinstance(command, str):
        cmd = shlex.split(command)
    else:
        cmd = list(command)

    # Validate all command arguments
    for arg in cmd:
        _validate_ros_arg(arg, "command argument")

    valid_ros2_commands = ["node", "topic", "service", "param", "doctor"]

    if len(cmd) < 2:
        raise ValueError(f"'{' '.join(cmd)}' is not a valid ROS2 command.")
    if cmd[0] != "ros2":
        raise ValueError(f"'{' '.join(cmd)}' is not a valid ROS2 command.")
    if cmd[1] not in valid_ros2_commands:
        raise ValueError(f"'ros2 {cmd[1]}' is not a valid ros2 subcommand.")

    try:
        output = subprocess.check_output(cmd, shell=False).decode()
        return True, output
    except Exception as e:
        return False, str(e)


def get_entities(
    cmd: str,
    delimiter: str = "\n",
    pattern: str = None,
    blacklist: Optional[List[str]] = None,
) -> List[str]:
    """
    Get a list of ROS2 entities (nodes, topics, services, etc.) via fallback subprocess path.
    """
    success, output = execute_ros_command(cmd)

    if not success:
        return [output]

    entities = output.split(delimiter)

    # Filter out blacklisted entities
    if blacklist:
        entities = list(
            filter(
                lambda x: not any(
                    re.match(f".*{pat}.*", x) for pat in blacklist
                ),
                entities,
            )
        )

    if pattern:
        entities = list(filter(lambda x: re.match(f".*{pattern}.*", x), entities))

    entities = [e for e in entities if e.strip() != ""]

    return entities


@tool
def ros2_node_list(pattern: Optional[str] = None, blacklist: Optional[List[str]] = None) -> dict:
    """
    Get a list of ROS2 nodes running on the system.

    :param pattern: A regular expression pattern to filter the list of nodes.
    """
    node = ROSANode.get_instance()
    if node is not None:
        try:
            node_names = node.get_node_names_and_namespaces()
            nodes = [f"{ns}/{name}".replace("//", "/") for name, ns in node_names]
            if blacklist:
                nodes = [n for n in nodes if not any(re.match(f".*{b}.*", n) for b in blacklist)]
            if pattern:
                nodes = [n for n in nodes if re.match(f".*{pattern}.*", n)]
            return {"nodes": nodes}
        except Exception:
            pass

    nodes = get_entities("ros2 node list", pattern=pattern, blacklist=blacklist)
    return {"nodes": nodes}


@tool
def ros2_topic_list(pattern: Optional[str] = None, blacklist: Optional[List[str]] = None) -> dict:
    """
    Get a list of ROS2 topics.

    :param pattern: A regular expression pattern to filter the list of topics.
    """
    node = ROSANode.get_instance()
    if node is not None:
        try:
            topic_names_and_types = node.get_topic_names_and_types()
            topics = [name for name, _ in topic_names_and_types]
            if blacklist:
                topics = [t for t in topics if not any(re.match(f".*{b}.*", t) for b in blacklist)]
            if pattern:
                topics = [t for t in topics if re.match(f".*{pattern}.*", t)]
            return {"topics": topics}
        except Exception:
            pass

    topics = get_entities("ros2 topic list", pattern=pattern, blacklist=blacklist)
    return {"topics": topics}


@tool
def ros2_topic_echo(
    topic: str,
    count: int = 1,
    return_echoes: bool = False,
    delay: float = 1.0,
    timeout: float = 1.0,
) -> dict:
    """
    Echoes the contents of a specific ROS2 topic.

    :param topic: The name of the ROS topic to echo.
    :param count: The number of messages to echo. Valid range is 1-10.
    :param return_echoes: If True, return the messages as a list with the response.
    :param delay: Time to wait between each message in seconds.
    :param timeout: Max time to wait for a message before timing out.
    """
    _validate_ros_arg(topic, "topic name")
    if count < 1 or count > 10:
        return {"error": "Count must be between 1 and 10."}

    cmd = ["ros2", "topic", "echo", topic, "--once", "--spin-time", str(timeout)]
    echoes = []
    for i in range(count):
        success, output = execute_ros_command(cmd)

        if not success:
            return {"error": output}

        print(output)
        if return_echoes:
            echoes.append(output)

        time.sleep(delay)

    if return_echoes:
        return {"echoes": echoes}

    return {"success": True}


@tool
def ros2_service_list(
    pattern: Optional[str] = None, blacklist: Optional[List[str]] = None
) -> dict:
    """
    Get a list of ROS2 services.

    :param pattern: A regular expression pattern to filter the list of services.
    """
    node = ROSANode.get_instance()
    if node is not None:
        try:
            service_names_and_types = node.get_service_names_and_types()
            services = [name for name, _ in service_names_and_types]
            if blacklist:
                services = [s for s in services if not any(re.match(f".*{b}.*", s) for b in blacklist)]
            if pattern:
                services = [s for s in services if re.match(f".*{pattern}.*", s)]
            return {"services": services}
        except Exception:
            pass

    services = get_entities("ros2 service list", pattern=pattern, blacklist=blacklist)
    return {"services": services}


@tool
def ros2_node_info(nodes: List[str]) -> dict:
    """
    Get information about a ROS2 node.

    :param nodes: A list of ROS2 node names.
    """
    node = ROSANode.get_instance()
    data = {}

    for node_name in nodes:
        _validate_ros_arg(node_name, "node name")
        if node is not None:
            try:
                parts = node_name.rsplit('/', 1)
                ns = parts[0] if parts[0] else "/"
                name = parts[1]

                pubs = node.get_publisher_names_and_types_by_node(name, ns)
                subs = node.get_subscriber_names_and_types_by_node(name, ns)
                srvs = node.get_service_names_and_types_by_node(name, ns)

                info = f"Node: {node_name}\n"
                info += "  Publishers:\n"
                for p_name, p_types in pubs:
                    info += f"    {p_name} [{', '.join(p_types)}]\n"
                info += "  Subscribers:\n"
                for s_name, s_types in subs:
                    info += f"    {s_name} [{', '.join(s_types)}]\n"
                info += "  Services:\n"
                for sv_name, sv_types in srvs:
                    info += f"    {sv_name} [{', '.join(sv_types)}]\n"

                data[node_name] = info
                continue
            except Exception:
                pass

        success, output = execute_ros_command(["ros2", "node", "info", node_name])
        if not success:
            data[node_name] = dict(error=output)
            continue
        data[node_name] = output

    return data


@tool
def ros2_topic_info(topics: List[str]) -> dict:
    """
    Get information about a ROS2 topic.

    :param topics: A list of ROS2 topic names.
    """
    node = ROSANode.get_instance()
    data = {}

    for topic in topics:
        _validate_ros_arg(topic, "topic name")
        if node is not None:
            try:
                pubs = node.get_publishers_info_by_topic(topic)
                subs = node.get_subscriptions_info_by_topic(topic)

                info = f"Topic: {topic}\n"
                info += f"  Publisher count: {len(pubs)}\n"
                info += f"  Subscriber count: {len(subs)}\n"
                data[topic] = info
                continue
            except Exception:
                pass

        success, output = execute_ros_command(["ros2", "topic", "info", topic, "--verbose"])
        if not success:
            topic_info = dict(error=output)
        else:
            topic_info = output

        data[topic] = topic_info

    return data


@tool
def ros2_param_list(
    node_name: Optional[str] = None,
    pattern: str = None,
    blacklist: Optional[List[str]] = None,
) -> dict:
    """
    Get a list of parameters for a ROS2 node.

    :param node_name: An optional ROS2 node name to get parameters for. If not provided, all parameters are listed.
    :param pattern: A regular expression pattern to filter the list of parameters.
    """
    if node_name:
        _validate_ros_arg(node_name, "node name")
        cmd = ["ros2", "param", "list", node_name]
        success, output = execute_ros_command(cmd)
        if not success:
            return {"error": output}

        params = [o for o in output.split("\n") if o]
        if pattern:
            params = [p for p in params if re.match(f".*{pattern}.*", p)]
        if blacklist:
            params = [
                p for p in params if not any(re.match(f".*{b}.*", p) for b in blacklist)
            ]
        return {node_name: params}
    else:
        cmd = ["ros2", "param", "list"]
        success, output = execute_ros_command(cmd)

        if not success:
            return {"error": output}

        lines = output.split("\n")
        data = {}
        current_node = None
        for line in lines:
            if line.startswith("/"):
                current_node = line
                data[current_node] = []
            elif line.strip() != "":
                data[current_node].append(line.strip())

        if pattern:
            data = {k: v for k, v in data.items() if re.match(f".*{pattern}.*", k)}
        if blacklist:
            data = {
                k: v
                for k, v in data.items()
                if not any(re.match(f".*{b}.*", k) for b in blacklist)
            }
        return data


@tool
def ros2_param_get(node_name: str, param_name: str) -> dict:
    """
    Get the value of a parameter for a ROS2 node.

    :param node_name: The name of the ROS2 node.
    :param param_name: The name of the parameter.
    """
    _validate_ros_arg(node_name, "node name")
    _validate_ros_arg(param_name, "parameter name")
    cmd = ["ros2", "param", "get", node_name, param_name]
    success, output = execute_ros_command(cmd)

    if not success:
        return {"error": output}

    return {param_name: output}


@tool
def ros2_param_set(node_name: str, param_name: str, param_value: str) -> dict:
    """
    Set the value of a parameter for a ROS2 node.

    :param node_name: The name of the ROS2 node.
    :param param_name: The name of the parameter.
    :param param_value: The value to set the parameter to.
    """
    _validate_ros_arg(node_name, "node name")
    _validate_ros_arg(param_name, "parameter name")
    _validate_ros_arg(str(param_value), "parameter value")
    cmd = ["ros2", "param", "set", node_name, param_name, str(param_value)]
    success, output = execute_ros_command(cmd)

    if not success:
        return {"error": output}

    return {param_name: output}


@tool
def ros2_service_info(services: List[str]) -> dict:
    """
    Get information about a ROS2 service.

    :param services: a list of ROS2 service names.
    """
    node = ROSANode.get_instance()
    data = {}

    for service_name in services:
        _validate_ros_arg(service_name, "service name")
        if node is not None:
            try:
                srv_names_and_types = node.get_service_names_and_types()
                srv_type = next((types for name, types in srv_names_and_types if name == service_name), None)
                if srv_type:
                    data[service_name] = srv_type[0]
                    continue
            except Exception:
                pass

        success, output = execute_ros_command(["ros2", "service", "type", service_name])
        if not success:
            data[service_name] = dict(error=output)
            continue
        data[service_name] = output

    return data


@tool
def ros2_service_call(service_name: str, srv_type: str, request: str) -> dict:
    """
    Call a ROS2 service.

    :param service_name: The name of the ROS2 service.
    :param srv_type: The type of the service (use ros2_service_info to verify).
    :param request: The request to send to the service.
    """
    _validate_ros_arg(service_name, "service name")
    _validate_ros_arg(srv_type, "service type")
    cmd = ["ros2", "service", "call", service_name, srv_type, request]
    success, output = execute_ros_command(cmd)
    if not success:
        return {"error": output}
    return {"response": output}


@tool
def ros2_topic_pub(
    topic: str,
    msg_type: str,
    message: str,
    rate: float = 10.0,
    duration: float = 1.0
) -> dict:
    """
    Publish a message to a ROS2 topic.

    :param topic: The name of the topic to publish to.
    :param msg_type: The message type (e.g. 'std_msgs/msg/String').
    :param message: The message content in YAML/JSON format.
    :param rate: The frequency (Hz) at which to publish the message. Defaults to 10.0 Hz.
    :param duration: The total duration (seconds) to keep publishing the message. Defaults to 1.0 second.
    """
    _validate_ros_arg(topic, "topic name")
    _validate_ros_arg(msg_type, "message type")
    
    node = ROSANode.get_instance()
    if node is not None:
        try:
            parts = msg_type.split('/')
            package = parts[0]
            module = parts[1] if len(parts) > 2 else "msg"
            cls_name = parts[-1]
            
            import importlib
            mod = importlib.import_module(f"{package}.{module}")
            msg_cls = getattr(mod, cls_name)
            
            import yaml
            msg_data = yaml.safe_load(message)
            
            from rosidl_runtime_py import set_message_fields
            msg = msg_cls()
            set_message_fields(msg, msg_data)
            
            pub = node.create_publisher(msg_cls, topic, 10)
            
            # Publish loop to maintain velocity/messages over duration
            sleep_time = 1.0 / max(0.1, rate)
            steps = int(duration * rate)
            for _ in range(max(1, steps)):
                pub.publish(msg)
                time.sleep(sleep_time)
            
            node.destroy_publisher(pub)
            return {"success": True, "message": f"Published to {topic} for {duration}s at {rate}Hz"}
        except Exception as e:
            pass

    # Fallback to subprocess using --rate and --times
    times = max(1, int(duration * rate))
    cmd = ["ros2", "topic", "pub", "--rate", str(rate), "--times", str(times), topic, msg_type, message]
    success, output = execute_ros_command(cmd)
    if not success:
        return {"error": output}
    return {"success": True, "output": output}


@tool
def ros2_doctor() -> dict:
    """
    Check ROS setup and other potential issues.
    """
    cmd = ["ros2", "doctor"]
    success, output = execute_ros_command(cmd)
    if not success:
        return {"error": output}
    return {"results": output}


def ros2_log_directories():
    """Get any available ROS2 log directories."""
    log_dir = get_logging_directory()
    print(f"ROS 2 logs are stored in: {log_dir}")

    return {"default": f"{log_dir}"}


@tool
def roslog_list(min_size: int = 2048, blacklist: Optional[List[str]] = None) -> dict:
    """
    Returns a list of ROS log files.

    :param min_size: The minimum size of the log file in bytes to include in the list.
    """
    logs = []
    log_dirs = ros2_log_directories()

    for _, log_dir in log_dirs.items():
        if not log_dir:
            continue

        log_files = [
            os.path.join(log_dir, f)
            for f in os.listdir(log_dir)
            if os.path.isfile(os.path.join(log_dir, f)) and f.endswith(".log")
        ]

        print(f"Log files: {log_files}")

        if blacklist:
            log_files = list(
                filter(
                    lambda x: not any(
                        re.match(f".*{pattern}.*", x) for pattern in blacklist
                    ),
                    log_files,
                )
            )

        log_files = list(filter(lambda x: os.path.getsize(x) > min_size, log_files))

        log_files = [
            {
                f.replace(log_dir, ""): (
                    f"{round(os.path.getsize(f) / 1024, 2)} KB"
                    if os.path.getsize(f) < 1024 * 1024
                    else f"{round(os.path.getsize(f) / (1024 * 1024), 2)} MB"
                ),
            }
            for f in log_files
        ]

        if len(log_files) > 0:
            logs.append(
                {
                    "directory": log_dir,
                    "total": len(log_files),
                    "files": log_files,
                }
            )

    return dict(
        total=len(logs),
        logs=logs,
    )
