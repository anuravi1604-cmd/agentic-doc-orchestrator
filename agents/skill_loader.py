"""
skill_loader.py
----------------
Loads and parses agent skill definitions (.md files) into structured runtime
guidelines and prompts for agents.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentSkill:
    name: str
    description: str
    when_to_use: str
    tools: List[str]
    procedure: List[str]
    guidelines: List[str]
    raw_markdown: str


class SkillLoader:
    """Discovers and loads agent skills from the skills directory."""

    def __init__(self, skills_dir: Optional[str] = None):
        if skills_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            skills_dir = os.path.join(base, "skills")
        self.skills_dir = skills_dir
        self._skills: Dict[str, AgentSkill] = {}
        self.load_all()

    def load_all(self) -> Dict[str, AgentSkill]:
        if not os.path.isdir(self.skills_dir):
            return {}

        for fname in os.listdir(self.skills_dir):
            if fname.endswith(".md"):
                fpath = os.path.join(self.skills_dir, fname)
                skill = self._parse_skill_file(fpath)
                if skill:
                    self._skills[skill.name] = skill
        return self._skills

    def get_skill(self, name: str) -> Optional[AgentSkill]:
        return self._skills.get(name)

    def _parse_skill_file(self, path: str) -> Optional[AgentSkill]:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract title
        title_match = re.search(r"^#\s*Skill:\s*(.+)$", content, re.MULTILINE)
        name = title_match.group(1).strip() if title_match else os.path.basename(path)

        # Extract description or when to use
        when_match = re.search(r"##\s*When to use[^\n]*\n([\s\S]*?)(?=\n##|\Z)", content)
        when_to_use = when_match.group(1).strip() if when_match else ""

        # Extract tools
        tools_match = re.search(r"##\s*Available tools[^\n]*\n([\s\S]*?)(?=\n##|\Z)", content)
        tools: List[str] = []
        if tools_match:
            for line in tools_match.group(1).splitlines():
                t_match = re.search(r"[-*]\s*`?([a-zA-Z0-9_\-]+)`?", line)
                if t_match:
                    tools.append(t_match.group(1))

        # Extract procedure steps
        proc_match = re.search(r"##\s*Procedure[^\n]*\n([\s\S]*?)(?=\n##|\Z)", content)
        procedure: List[str] = []
        if proc_match:
            for line in proc_match.group(1).splitlines():
                step_match = re.search(r"^\d+\.\s*(.+)$", line.strip())
                if step_match:
                    procedure.append(step_match.group(1))

        # Extract notes/guidelines
        notes_match = re.search(r"##\s*Notes[^\n]*\n([\s\S]*?)(?=\n##|\Z)", content)
        guidelines: List[str] = []
        if notes_match:
            for line in notes_match.group(1).splitlines():
                line = line.strip().lstrip("-* ").strip()
                if line:
                    guidelines.append(line)

        return AgentSkill(
            name=name,
            description=when_to_use.split("\n")[0] if when_to_use else name,
            when_to_use=when_to_use,
            tools=tools,
            procedure=procedure,
            guidelines=guidelines,
            raw_markdown=content,
        )
