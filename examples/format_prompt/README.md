# Format Prompt Templates

This directory contains Jinja2 templates for formatting prompts in RLHF datasets.

## Overview

The format prompt feature allows you to apply custom formatting to each prompt in your dataset using Jinja2 templates. This is useful when you want to add consistent instructions or formatting to all prompts without modifying the original dataset.

## Default Template

The default template (`default.jinja`) appends the following instruction to each prompt:

```
{{ content }}You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.
```

## Usage

To use a format prompt template, specify the `format_prompt` parameter in your data configuration:

```yaml
data:
  # ... other data config ...
  format_prompt: examples/format_prompt/default.jinja  # Path to your template file
```

Or set it to `null` to disable format prompting:

```yaml
data:
  format_prompt: null
```

## Creating Custom Templates

To create a custom format prompt:

1. Create a new `.jinja` file in this directory or elsewhere
2. Use `{{ content }}` as the placeholder for the original prompt content
3. Add your custom formatting around it

Example custom template:

```jinja
{{ content }}

Please solve this problem step by step:
1. Understand the problem
2. Plan your approach
3. Execute the solution
4. Verify your answer
```

## Template Variables

Currently, the template receives one variable:
- `content`: The original prompt text

## Notes

- The template is applied during dataset preprocessing
- If the template file is not found, the system will use the original prompt without formatting
- For multimodal datasets (images/videos), the formatting is applied to text segments only