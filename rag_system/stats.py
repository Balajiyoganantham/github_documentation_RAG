import statistics

def get_conversation_stats(conversation_history, confidence_scores, response_times, memory):
    if not conversation_history:
        return {}
    return {
        "total_questions": len(conversation_history),
        "avg_confidence": statistics.mean(confidence_scores) if confidence_scores else 0,
        "avg_response_time": statistics.mean(response_times) if response_times else 0,
        "memory_size": len(memory.chat_memory.messages),
        "unique_sources": len(set([
            source for conv in conversation_history 
            for source in conv.get('sources', [])
        ]))
    }

def get_memory_summary(memory):
    if not memory or not memory.chat_memory.messages:
        return "No conversation history"
    messages = memory.chat_memory.messages
    return f"Conversation contains {len(messages)} messages. Recent topics discussed: {', '.join([msg.content[:50] + '...' for msg in messages[-3:] if hasattr(msg, 'content')])}" 