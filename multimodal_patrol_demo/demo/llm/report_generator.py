from typing import List

from demo.llm.llm_client import BaseLLMClient
from demo.types import AlertEvent, PatrolReport


class ReportGenerator:
    def __init__(self, llm_client: BaseLLMClient) -> None:
        self.llm_client = llm_client

    def build_event_prompt(self, event: AlertEvent) -> str:
        return (
            f"时间：{event.timestamp:.1f} 秒，"
            f"在区域 {event.zone_name} 发现 {event.class_name}，"
            f"距离约 {event.distance_m:.1f} 米，"
            f"在区域内停留约 {event.duration_s:.1f} 秒。"
        )

    def describe_single_event(self, event: AlertEvent) -> str:
        system_prompt = (
            "你是一个智能巡检系统，需要用简洁、专业的中文描述巡检事件。"
            "不要使用第一人称，不要添加未给出的信息。"
        )
        user_prompt = (
            "根据以下结构化信息，生成一句话的事件描述：\n"
            + self.build_event_prompt(event)
        )
        return self.llm_client.generate(system_prompt, user_prompt)

    def summarize_report(self, report: PatrolReport) -> PatrolReport:
        if not report.events:
            report.summary_text = "本次巡检过程中未发现明显异常事件。"
            return report

        lines: List[str] = [self.build_event_prompt(e) for e in report.events]
        events_block = "\n".join(lines)

        system_prompt = (
            "你是一个智能巡检系统，需要根据多条事件记录生成一段巡检总结。"
            "输出简洁中文，不要超过 120 字。"
        )
        user_prompt = (
            "以下是本次巡检期间记录的事件，请给出总体巡检总结：\n" + events_block
        )
        summary = self.llm_client.generate(system_prompt, user_prompt)
        report.summary_text = summary
        return report
