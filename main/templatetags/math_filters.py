from django import template

register = template.Library()

@register.filter
def mul(value, arg):
    """
    템플릿에서 값을 곱하는 필터.
    사용법: {{ some_value|mul:100 }}
    """
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return ''