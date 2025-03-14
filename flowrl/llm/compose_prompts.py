import agentkit.compose_prompt as cp 

class ComposeBasePrompt(cp.ComposePromptDB):

    def __init__(self, system_prompt=""):
        super().__init__()
        self.shrink_idx = 1

    def before_dependencies(self, messages, db):
        return messages

    def after_dependencies(self, messages, db):
        return messages

    def compose(self, dependencies, prompt):

        msg = [{"role": "user", "content": self.system_prompt}]

        msg = self.before_dependencies(msg, self.node.db)

        msg = self.add_dependencies(msg, dependencies, self.node.db)

        msg = self.after_dependencies(msg, self.node.db)

        prompt, db_retrieval_results = self.render_db(prompt, self.node.db)
        self.node.rendered_prompt = prompt
        self.node.db_retrieval_results = db_retrieval_results

        msg.append({"role": "user", "content": prompt})

        return msg, self.shrink_idx

class ComposeReasoningPrompt(ComposeBasePrompt):
    
    def __init__(self):
        super().__init__()
        self.system_prompt = "Make the best plan for the game by analyzing the instruction manual. The game does not contain bugs or glitches, and does not offer additional cues or patterns."

    def before_dependencies(self, messages, db):
        messages.append({"role": "user", "content": "Instruction manual:\n\n{}".format(db['manual'])})
        return messages