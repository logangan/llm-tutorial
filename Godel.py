# @misc{peng2022godel,
# author = {Peng, Baolin and Galley, Michel and He, Pengcheng and Brockett, Chris and Liden, Lars and Nouri, Elnaz and Yu, Zhou and Dolan, Bill and Gao, Jianfeng},
# title = {GODEL: Large-Scale Pre-training for Goal-Directed Dialog},
# howpublished = {arXiv},
# year = {2022},
# month = {June},
# url = {https://www.microsoft.com/en-us/research/publication/godel-large-scale-pre-training-for-goal-directed-dialog/},
# }
#
# Multi-turn generation examples from an interactive environment:
# Chitchat example:
#
# Instruction: given a dialog context, you need to response empathically.
# User: Does money buy happiness?
# Agent: It is a question. Money buys you a lot of things, but not enough to buy happiness.
# User: What is the best way to buy happiness ?
# Agent: Happiness is bought through your experience and not money.
#
# Grounded response generation example:
#
# Instruction: given a dialog context and related knowledge, you need to response safely based on the knowledge.
# Knowledge: The best Stardew Valley mods PCGamesN_0 / About SMAPI
# User: My favorite game is stardew valley. stardew valley is very fun.
# Agent: I love Stardew Valley mods, like PCGamesN_0 / About SMAPI.


import sys, time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def print_output(text):
    count = 0
    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()
    for a in text:
        sys.stdout.write(a)
        sys.stdout.flush()
        time.sleep(0.01)
        count += 1
        if count % 200 == 0:
            print(end='\n')


tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")


def generate(instruction, knowledge, dialog):
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=50, min_length=20, top_p=0.5, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output


# chitchat task
# instruction1 = f'Instruction: given a dialog context, you need to response empathically.'
instruction1 = f'Instruction: Translate the content to Chinese'
# Leave the knowledge empty

# Grounded Response Generation
instruction2 = f'Instruction: given a dialog context and related knowledge, you need to response safely ' \
               f'based on the knowledge.'
# instruction2 = f'Instruction: given a dialog context and related knowledge, you need to response safely ' \
#                f'in Chinese language based on the knowledge.'

# Conversational Question Answering
instruction3 = f'Instruction: given a dialog context and related knowledge, you need to answer the question' \
               f' based on the knowledge.'

# f = open('data\IT_knowledge.txt', 'r', encoding='utf-8')
# knowledge1 = f.readlines()
# knowledge2 = ",".join(knowledge1)
# # print(len(knowledge2))
# knowledge = "Robbie is the chatbot service, can help users to install software, printers, reset password, and " \
#             "provide fast Q&A to answer users questions on IT, HR, Admin, Finance. ITsupport is the agent service, " \
#             "can help on all kinds of issues reported to them. The fast way is to ask Robbie for help, as ITsupport" \
#             " will need to wait for an available agent to help you out, which will almost need 30 minutes while " \
#             "Robbie only needs seconds to help you out. The Robbie can be reach via ss.lenovo.com. ITsupport can be" \
#             "reached via itsupport.lenovo.com or ITsupport on Teams."
knowledge = ''
dialog1 = [
    # 'user: PROBLEMS TO ACCESS SAP EGP and PROBLEMS TO INSTALL MS VISIO'
    # 'agent: Supported user to unlock SAP by ITSUPPORT Portal / Downloaded and INstalled Visio from MS portal'
    # 'user: User reported that the printer was not working. '
    # 'agent: I checked the print list and there were several files in the list that were holding up the printer. '
    # 'Then when I deleted all the files, it made a new attempt and worked perfectlyTicket solved.'
    # 'user: User reported that Teams was not working notifications. '
    # 'agent: I instructed him to fix any problems by doing a manual update within Teams itself. Com teams downloaded '
    # 'the latest version, restarted after installation. And the user infomouiated that it worked perfectly again.'
    # 'how to install printer? please think it step by step'
    'how to contact hrsupport for help?'
]

dialog2 = [
    'user: I have password reset request'
    'agent: I would like to help, you can go to robbie for help.The link is ss.lenovo.com for reference'
    'I want to reset my password but I could not know how, can you help me?'
]

dialog3 = [
    '如何安装系统'
    '你可以通过自助服务或者机器人获取帮助'
    '我要重装系统，如何操作？'
]

dialog4 = [
    'what\'s the results of 2 plus'
    'what\'s the results of 2 plus 64 times.'

]

dialog5 = [
    # 'user: what if I want to install software?'
    # 'agent: the link is ss.lenovo.com'
    'I can not log in windows, how to do? could you help on the password?'
]

for i in range(20):
    response = generate(instruction1, knowledge, dialog3)
    # count = 0
    print('\n', i, ')')
    print_output(str(response))

    # for j in range(len(response)):
    #     print(str(response)[j], end='')
    #     count += 1
    #     if count % 200 == 0:
    #         print(end='\n')
    i += 1
