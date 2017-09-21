#!/usr/local/bin/python3
# Original Message Handler of the Chatbot Webhook APP
# Created By Intralogue

from json import JSONDecodeError
from re import sub

from requests import post as req_post

from chatbot.correction.spell_check import spell_checker

from chatbot.nlp.memm import BookSlot_MEMM
from chatbot.nlp.intent_classifier import predict

from chatbot.helper import debug


def parser(original):
    debug(original, 'Parse Original Message:')
    # message = spell_checker(original)

    # if message:
    message = [original]

    debug('Message After spell check: {}'.format(message))
    tmpslot = dict(
        originalText=[],
        lemma=[],
        pos=[],
        slot_tag=[],
    )

    try:
        slot_response = req_post(
            'http://13.228.72.161:8080/tag',
            json={
                "q": message
            }
        ).json()
        general_slots = slot_response.get('result') or [('LEXICA_SLOT',) * 4]
        debug('general_slots: {}'.format(general_slots))
        (
            tmpslot['originalText'],
            tmpslot['lemma'],
            tmpslot['pos'],
            tmpslot['slot_tag']
        ) = zip(*general_slots)
    except JSONDecodeError as err:
        debug('JSON error [{}]'.format(err), 'ERROR')

    message = ' '.join(message)
    predict_result = dict(
        zip(('label', 'original', 'stanford'), predict(message))
    )
    debug(predict_result, name='Predict Result:')
    # predict_result['stanford'] = predict_result.get('stanford')[0]

    label_score = float(predict_result.get('label')[0].get('score'))
    original_score = float(predict_result.get('original')[0].get('score'))

    if label_score > original_score:
        class_probability = predict_result.get('label')
    else:
        class_probability = predict_result.get('original')

    del label_score, original_score, predict_result

    words = list(filter(None, [
        word.strip()
        for word in sub(r'[^\w\s]', ' ', message).split()
    ]))

    # create new words for BookSlot_MEMM prevent mutation of words
    book_slot = BookSlot_MEMM(words[:])

    general_slots = list(map(list, zip(*list(tmpslot.values()))))
    book_keyword = ''
    for i, val in enumerate(words):
        if book_slot[i] == 'book':
            book_keyword += ' ' + val

    book_keyword = book_keyword.strip() if book_keyword.strip() else message

    debug(book_slot, name='book_slot: ')
    debug(book_keyword, name='book_keyword: ')
    debug(class_probability, name='class_probability: ')
    debug(general_slots, name='general_slots: ')
    debug(words, name='words: ')

    return message, book_keyword, class_probability, general_slots, words
