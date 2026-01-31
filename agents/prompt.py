import json

def build_agent_prompt(perception_stats: dict, agent_info: dict) -> str:
    """
    Build the prompt for the agent model inference.
    
    Args:
        perception_stats: Dictionary containing statistics about the perceived area
        agent_info: Dictionary containing agent attributes
    
    Returns:
        Formatted prompt string for Claude
    """
    prompt = f"""You are one of many independent agents interacting with a shared canvas.

You do not know the final image and you are not told what to draw.

You only observe:
- The current colours in your local region
- The colours of neighbouring regions
- A small set of agent attributes that reflect your local preferences

Your goal is to locally improve coherence and structure by adjusting your region in response to what you observe.

Act under these principles:
- Encourage smooth transitions where neighbouring colours are similar
- Preserve or sharpen boundaries where strong contrasts already exist
- Reinforce stable patterns that persist across iterations
- Avoid introducing new structure unless local evidence supports it
- Do not attempt to impose global patterns or symbols
- Assume other agents are acting under the same rules

Perception statistics of surrounding area:
{json.dumps(perception_stats, indent=2)}

Agent attributes (your local preferences):
{json.dumps(agent_info, indent=2)}

Based on your observations, propose a small colour update for your region that you believe improves local consistency with the evolving canvas.

Provide:
1. An RGB color (values 0-255) that balances coherence with the surrounding area while respecting your stylistic biases
2. A confidence score (0-1) indicating how confident you are in this choice

Consider:
- The mean and std of surrounding colors when deciding smoothness
- Your stylistic biases (contrast, smoothness, edge detection)
- Your temperature parameter (higher = more variation, lower = more conservative)

Respond in JSON format only:
{{
    "r": <0-255>,
    "g": <0-255>,
    "b": <0-255>,
    "confidence": <0.0-1.0>
}}"""
    return prompt
