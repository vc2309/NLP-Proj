#!/usr/local/bin/python3
# DynamoDB Related
# Created By Intralogue
# Version 2.0.0
from json import loads

from chatbot.helper import debug, clean_empty, NONE

# AWS lib
from boto3 import resource


class Model:
    _dynamodb = resource('dynamodb', region_name='ap-southeast-1')

    def __init__(self, rid, sid, university):
        self._rid = int(rid)
        self._sid = int(sid)
        self._history = self._dynamodb.Table('-'.join(
            ['lexica', university, 'history']
        ))
        self._memories = self._dynamodb.Table('-'.join(
            ['lexica', university, 'memories']
        ))
        self._resources = self._dynamodb.Table('-'.join(
            ['lexica', university, 'resources']
        ))
        self._result = self._dynamodb.Table('-'.join(
            ['lexica', university, 'result']
        ))
        self._sender = self._dynamodb.Table('-'.join(
            ['lexica', university, 'sender']
        ))

        # item = dict(sender=int(sid))
        # exist = self._sender.get_item(Key=item)
        # if not exist:
        #     self._sender.put_item(Item=item)

        self.memory = None
        self.get_memory()

    def get_memory(self):
        if not self.memory:
            result = self._memories.get_item(Key=dict(sender=self._sid))
            item = result.get('Item', {})
            query = item.get('cquery', {})

            self.memory = dict(
                context=item.get('context', NONE),
                cquery=dict(
                    keyword=query.get('keyword', NONE),
                    refinements=query.get('refinements', []),
                    authors=query.get('authors', []),
                    date=query.get('date', [])
                )
            )

        debug(self.memory)
        return self.memory

    def upload_memory(self):
        if isinstance(self.memory, str):
            self.memory = loads(self.memory)
            if not isinstance(self.memory, dict):
                self.memory = {}

        memory = self.memory.copy()
        cquery = memory.get('cquery', {})
        self.memory = dict(
            context=memory.get('context', NONE) or NONE,
            cquery=dict(
                keyword=cquery.get('keyword', NONE) or NONE,
                refinements=list(set(cquery.get('refinements', []) or [])),
                authors=cquery.get('authors', []) or [],
                date=cquery.get('date', []) or [],
            ),
            sender=int(self._sid)
        )

        # debug(self.memory, 'Before Sanitized Memory:')
        self.memory = clean_empty(self.memory)
        # debug(self.memory, 'After Sanitized Memory:')

        self._memories.put_item(Item=self.memory)
        debug('Updated memory on dynamodb.')

        debug(self.memory, 'DynamoDB memory:')

    def get_payload(self, payload):
        return dict(
            action=loads(self._resources.get_item(Key={
                'type': 'payload',
                'key': str(payload)
            }).get('Item', {}).get('value', '{}')),
            payload=payload
        )

    def get_response(self, intent):
        debug(intent, 'geting response for intent:')
        return self._resources.get_item(Key={
            'type': 'responses',
            'key': str(intent)
        }).get('Item', {}).get('value')

    def search_result(self, resultobj):
        debug('creating new search result')
        for result in resultobj:
            tmp = clean_empty(result.copy())
            self._result.put_item(Item=dict(
                image=tmp.get('image', NONE),
                subtitle=tmp.get('subtitle', NONE),
                summary=tmp.get('summary', NONE),
                title=tmp.get('title', NONE),
                url=tmp.get('url', NONE),
                id=tmp.get('uuid', NONE)
            ))

    def create_cache(self, cache):
        debug((
            'sender({sender}) datetime({datetime}) '
            'message({message}) original message({original_message})'
        ).format(**cache), 'create new cache')
        self._history.put_item(Item=clean_empty(cache))
