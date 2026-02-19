"""通用工具"""
from datetime import datetime
from langchain.tools import tool


@tool
def get_current_time() -> str:
    """获取当前世界时间，包括UTC时间和多个时区的时间。"""
    now = datetime.now()
    
    # 格式化输出
    result = f"""当前时间信息：

【本地时间】
{now.strftime("%Y年%m月%d日 %H:%M:%S %A")}
ISO格式: {now.isoformat()}

【UTC时间】
UTC: {now.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}

【时间戳】
Unix时间戳: {int(now.timestamp())}

【今日信息】
- 年份: {now.year}
- 月份: {now.month}
- 日期: {now.day}
- 星期: {now.strftime("%A")}
- 是闰年: {'是' if ((now.year % 4 == 0 and now.year % 100 != 0) or (now.year % 400 == 0)) else '否'}
"""
    return result


@tool
def get_stock_market_status() -> str:
    """获取当前中国股市交易状态判断（基于时间判断，非实时接口）。"""
    now = datetime.now()
    weekday = now.weekday()
    hour = now.hour
    minute = now.minute
    
    # 周末
    if weekday >= 5:
        return f"当前为周末（{['周一','周二','周三','周四','周五','周六','周日'][weekday]}），中国股市休市。"
    
    # 交易时段判断
    morning_open = 9
    morning_close = 11
    morning_close_minute = 30
    afternoon_open = 13
    afternoon_close = 15
    
    if hour < morning_open:
        return f"当前时间 {now.strftime('%H:%M')}，股市尚未开盘（上午 9:30 开盘）。"
    elif hour == morning_open and minute < 30:
        return f"当前时间 {now.strftime('%H:%M')}，股市即将开盘（上午 9:30 开盘）。"
    elif hour < morning_close or (hour == morning_close and minute <= morning_close_minute):
        return f"当前时间 {now.strftime('%H:%M')}，股市上午交易时段进行中。"
    elif hour < afternoon_open:
        return f"当前时间 {now.strftime('%H:%M')}，股市午间休市（下午 13:00 开盘）。"
    elif hour < afternoon_close:
        return f"当前时间 {now.strftime('%H:%M')}，股市下午交易时段进行中。"
    elif hour == afternoon_close and minute == 0:
        return f"当前时间 {now.strftime('%H:%M')}，股市刚刚收盘。"
    else:
        return f"当前时间 {now.strftime('%H:%M')}，股市已收盘（下午 15:00 收盘）。"
