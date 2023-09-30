"""Extract span alignments

Usage:
    extract_span_alignments.py DATA_PATH

"""

import pandas as pd
import ast
import math
from typing import List, Tuple, Dict, Union
from typing import TypeVar, Callable
from docopt import docopt
from pathlib import Path
import re

args = docopt(__doc__)

from dataclasses import dataclass

T = TypeVar('T')

def tokenize_target(target_string):

    pattern = r'(?<!id)\((?!all)'
    processed_string = re.split(pattern, target_string)
    has_opening_parenthesis = any('(' in s for s in processed_string)

    target_tokens = []

    if has_opening_parenthesis:
        for s in processed_string:
            if '(' in s:  
                if 'id' in s:
                    if 'name' in s:
                        if '_, ' in s:
                            split_tokens = re.split(r'(?<=_), ', s)
                            for i in range(len(split_tokens)):
                                s = split_tokens[i]
                                target_tokens.append(s + ')') if i==0 else target_tokens.append(s)
                        elif 'all' in s:
                            split_tokens = s.split(', ')
                            for s in split_tokens:
                                target_tokens.append(s + ')')
                        elif 'name, ' in s and s.endswith('_'):
                            target_tokens.append(s + ')')
                        elif 'name, ' in s and s.endswith('name'):
                            split_tokens = re.split(r'(?<=name), |(?<=id)\(', s)
                            target_tokens.append(s + ')')
                            for s in split_tokens:
                                if 'id' not in s:
                                    target_tokens.append(s)
                        elif 'name, ' in s: 
                            split_tokens = re.split(r'(?<=name), ', s)
                            for i in range(len(split_tokens)):
                                s = split_tokens[i]
                                target_tokens.append(s + ')') if i==0 else target_tokens.append(s)  
                        
                        else: target_tokens.append(s + ')')
                    elif ', ' in s:
                        split_tokens = s.split(', ')
                        for s in split_tokens:
                            if 'id' in s:
                                target_tokens.append(s + ')')
                            else: target_tokens.append(s)
                    
                    else:    
                        target_tokens.append(s + ')')       
                elif 'all' in s:
                        if ',' in s:
                            split_tokens = s.split(', ')
                            for s in split_tokens:
                                if 'all' in s:
                                    target_tokens.append(s + ')')
                            target_tokens.extend(split_tokens)  
                        else: target_tokens.append(s + ')')    
            else:
                target_tokens.append(s)
    else:
        target_tokens = processed_string

    return target_tokens


def euclidean_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def max_by(elements: List[T], key: Callable[[T], float]) -> T:
    best_element_pos = None
    best_element_score = float('-inf')

    if len(elements) == 0:
        raise ValueError('Size of list is zero')

    for i, element in enumerate(elements):
        score_i = key(element)
        if score_i > best_element_score:
            best_element_pos = i
            best_element_score = score_i

    return elements[best_element_pos]

@dataclass
class SpanAlignment(object):
    source_span: Tuple[int, int]
    target_span: Tuple[int, int]

    @property
    def centroid(self) -> Tuple[float, float]:
        return (
            self.source_span[0] + (self.source_span[1] - self.source_span[0]) / 2,
            self.target_span[0] + (self.target_span[1] - self.target_span[0]) / 2
        )

    def distance(self, other: 'SpanAlignment') -> float:
        left = other.source_span[1] < self.source_span[0]
        right = self.source_span[1] < other.source_span[0]
        bottom = self.target_span[1] < other.target_span[0]
        top = other.target_span[1] < self.target_span[0]

        if top and left:
            return euclidean_dist((self.source_span[0], self.target_span[0]), (other.source_span[1], other.target_span[1]))
        elif left and bottom:
            return euclidean_dist((self.source_span[0], self.target_span[1]), (other.source_span[1], other.target_span[0]))
        elif bottom and right:
            return euclidean_dist((self.source_span[1], self.target_span[1]), (other.source_span[0], other.target_span[0]))
        elif right and top:
            return euclidean_dist((self.source_span[1], self.target_span[0]), (other.source_span[0], other.target_span[1]))
        elif left:
            return self.source_span[0] - other.source_span[1]
        elif right:
            return other.source_span[0] - self.source_span[1]
        elif bottom:
            return other.target_span[0] - self.target_span[1]
        elif top:
            return self.target_span[0] - other.target_span[1]
        else:
            return 0.

    def source_distance(self, other: 'SpanAlignment') -> float:
        left = other.source_span[1] < self.source_span[0]
        right = self.source_span[1] < other.source_span[0]
        bottom = self.target_span[1] < other.target_span[0]
        top = other.target_span[1] < self.target_span[0]

        if left:
            return math.fabs(self.source_span[0] - other.source_span[1])
        elif right:
            return math.fabs(self.source_span[1] - other.source_span[0])
        else:
            return 0.

    def overlaps(self, other: 'SpanAlignment') -> bool:
        raise Exception('Buggy!')
        return (
            (
                self.source_span[0] <= other.source_span[0] <= self.source_span[1] or
                self.source_span[0] <= other.source_span[1] <= self.source_span[1]
            ) and
            (
                self.target_span[0] <= other.target_span[0] <= self.target_span[1] or
                self.target_span[1] <= other.target_span[1] <= self.target_span[1]
            )
        )

    def source_overlap(self, other: 'SpanAlignment') -> bool:
        return (
            self.source_span[0] <= other.source_span[0] <= self.source_span[1] - 1 or
            self.source_span[0] <= other.source_span[1] - 1 <= self.source_span[1] - 1 or
            other.source_span[0] <= self.source_span[0] and self.source_span[1] <= other.source_span[1]
        )

@dataclass
class MergedSpanAlignment(SpanAlignment):
    spans: List[SpanAlignment]
    nested_spans: List[Union[SpanAlignment, 'MergedSpanAlignment']]

    def __init__(self, spans):
        flattened_spans = []
        nested_spans = []
        for span in spans:
            nested_spans.append(span)

            if isinstance(span, MergedSpanAlignment):
                for atomic_span in span.spans:
                    flattened_spans.append(atomic_span)
            else:
                flattened_spans.append(span)

        self.spans = flattened_spans
        self.nested_spans = nested_spans

        source_span_start = min(s.source_span[0] for s in spans)
        source_span_end = max(s.source_span[1] for s in spans)
        target_span_start = min(s.target_span[0] for s in spans)
        target_span_end = max(s.target_span[1] for s in spans)

        self.source_span = (source_span_start, source_span_end)
        self.target_span = (target_span_start, target_span_end)

    def maximal_adjacent_descendant_span_source_distance(self) -> float:
        if len(self.spans) <= 1:
            return 0.

        max_dist = float('-inf')
        for span in self.spans:
            other_spans = [
                s for s in self.spans
                if s != span
            ]

            adj_other_span = max_by(other_spans, key=lambda other: -span.source_distance(other))
            dist = span.source_distance(adj_other_span)
            if max_dist < dist:
                max_dist = dist

        return max_dist

    @staticmethod
    def get_minimal_child_span_distance(span_1, span_2):
        spans_1 = span_1.spans if isinstance(span_1, MergedSpanAlignment) else [span_1]
        spans_2 = span_2.spans if isinstance(span_2, MergedSpanAlignment) else [span_2]

        return min(
            s1.distance(s2)
            for s1 in spans_1
            for s2 in spans_2
        )

def construct_span_alignments(
    source_tokens: List[str],
    target_tokens: List[str],
    token_alignment: List[Tuple[int, int]],
    merge: bool = False
):
    """
    Args:
        token_alignment: For each source token, its list of aligned tokens in the target side
    """
    span_alignments: List[SpanAlignment] = []
    merged_alignments: List[SpanAlignment] = []

    #filtered_token_alignment = [tup for tup in token_alignment if all(elem is not None for elem in tup)]

    token_alignment_set = token_alignment
    visited = {align: False for align in token_alignment_set}

    def _get_consecutive_span_alignment(src_idx, tgt_idx):  # noqa
        cur_src_start, cur_src_end = src_idx, src_idx + 1
        cur_tgt_start, cur_tgt_end = tgt_idx, tgt_idx + 1

        visited[(src_idx, tgt_idx)] = True

        def _is_visited(x, y):
            return visited[(x, y)]

        if (src_idx - 1, tgt_idx) in token_alignment_set and not _is_visited(src_idx - 1, tgt_idx):
            (new_src_start, new_src_end), (new_tgt_start, new_tgt_end) = _get_consecutive_span_alignment(src_idx - 1, tgt_idx)

            cur_src_start = min(cur_src_start, new_src_start)
            cur_src_end = max(cur_src_end, new_src_end)
            cur_tgt_start = min(cur_tgt_start, new_tgt_start)
            cur_tgt_end = max(cur_tgt_end, new_tgt_end)

        if (src_idx + 1, tgt_idx) in token_alignment_set and not _is_visited(src_idx + 1, tgt_idx):
            (new_src_start, new_src_end), (new_tgt_start, new_tgt_end) = _get_consecutive_span_alignment(src_idx + 1, tgt_idx)

            cur_src_start = min(cur_src_start, new_src_start)
            cur_src_end = max(cur_src_end, new_src_end)
            cur_tgt_start = min(cur_tgt_start, new_tgt_start)
            cur_tgt_end = max(cur_tgt_end, new_tgt_end)

        if (src_idx, tgt_idx - 1) in token_alignment_set and not _is_visited(src_idx, tgt_idx - 1):
            (new_src_start, new_src_end), (new_tgt_start, new_tgt_end) = _get_consecutive_span_alignment(src_idx, tgt_idx - 1)

            cur_src_start = min(cur_src_start, new_src_start)
            cur_src_end = max(cur_src_end, new_src_end)
            cur_tgt_start = min(cur_tgt_start, new_tgt_start)
            cur_tgt_end = max(cur_tgt_end, new_tgt_end)

        if (src_idx, tgt_idx + 1) in token_alignment_set and not _is_visited(src_idx, tgt_idx + 1):
            (new_src_start, new_src_end), (new_tgt_start, new_tgt_end) = _get_consecutive_span_alignment(src_idx, tgt_idx + 1)

            cur_src_start = min(cur_src_start, new_src_start)
            cur_src_end = max(cur_src_end, new_src_end)
            cur_tgt_start = min(cur_tgt_start, new_tgt_start)
            cur_tgt_end = max(cur_tgt_end, new_tgt_end)

        return (cur_src_start, cur_src_end), (cur_tgt_start, cur_tgt_end)

    for (src_idx, tgt_idx) in token_alignment_set:
        if not visited[(src_idx, tgt_idx)]:
            source_span, target_span = _get_consecutive_span_alignment(src_idx, tgt_idx)

            span_alignments.append(SpanAlignment(source_span, target_span))

    span_alignments = sorted(span_alignments, key=lambda a: (a.source_span[0], a.target_span[0]))

    if merge:
        # merge adjacent bounding boxes
        def _merge_neighboring_spans(span_1: SpanAlignment, span_2: SpanAlignment):
            merged_span = MergedSpanAlignment([span_1, span_2])
            return merged_span

        while True:
            # sort by source index
            span_alignments = sorted(span_alignments, key=lambda a: (a.source_span[0], a.target_span[0]))
            can_merge = False

            #print(span_alignments)

            for i in range(len(span_alignments) - 1):
                span = span_alignments[i]
                next_span = span_alignments[i + 1]
                #print('Current span: '+ str(span))
                #print(i)
                
                if next_span.target_span == (-1, 0) and span.target_span!=(-1,0):
                    if i==len(span_alignments) - 2:
                        next_span.target_span = span.target_span
                        can_merge=True
                    else:
                        merged_alignments.append([span.source_span, span.target_span])
                        continue

                if span.target_span==(-1,0):
                    can_merge = True
                    #if i == len(span_alignments) - 2:                              
                    #    can_merge_left = True          #Will make sense with dist bw word embeddings not indices
                    #span.source_span
                    span.target_span = next_span.target_span

                dist2 = span.distance(next_span)
                if dist2 <= 1 or dist2 <= math.sqrt(2):
                    can_merge = True

                    if (
                        isinstance(span, MergedSpanAlignment) or
                        isinstance(next_span, MergedSpanAlignment)
                    ):
                        
                        pairwise_dist = MergedSpanAlignment.get_minimal_child_span_distance(span, next_span)
                        if pairwise_dist > math.sqrt(2):
                            can_merge = False

                    if can_merge:
                        merged_span = _merge_neighboring_spans(span, next_span)

                        del span_alignments[i: i + 2]
                        merged_alignments.append([merged_span.spans,])

                        break

            if not can_merge:
                break

    return merged_alignments

def get_span_alignments(index, alignment_string, source_string, target_string):
    print(index)
    alignment_pairs = ast.literal_eval("[" + alignment_string + "]")
    #source_tokens = source_string.split()                          #Deal with city_name occurrences (clean data?)
    source_tokens = [pair[0] for pair in alignment_pairs]
    target_tokens = tokenize_target(target_string.replace(")", ""))
    print('Alignment pairs: '+str(alignment_pairs))

    print("\nSource tokens: "+str(source_tokens))
    print("Target tokens: " +str(target_tokens))
    
    token_alignments = []

    src_token_to_idx = {token: idx for idx, token in enumerate(source_tokens)}
    tgt_token_to_idx = {token: idx for idx, token in enumerate(target_tokens)}

    print(src_token_to_idx)
    print(tgt_token_to_idx)

    for src_token, tgt_token in alignment_pairs:
        #print(src_token)
        #print(tgt_token)
        src_idx = src_token_to_idx[src_token] if src_token!= 'ε' else -1
        tgt_idx = tgt_token_to_idx[tgt_token] if tgt_token!= 'ε' else -1

        token_alignments.append((src_idx, tgt_idx))
    print("Token alignments: " +str(token_alignments))

    span_alignments = construct_span_alignments(source_tokens, target_tokens, token_alignments, merge=True)
    print("\nCorresponding Span alignments: "+str(span_alignments))
    return span_alignments


def get_alignments(file_path):
    df = pd.read_csv(file_path)
    
    df = df.iloc[:, 1:]
    df['span_alignments'] = df.apply(lambda x: get_span_alignments(x['ID'], x['ALIGNMENT'], x['NL'], x['MR']), axis=1)
    data_list = df.to_dict(orient='records')
    
    return data_list

if __name__ == '__main__':
    span_lvl_alignments = get_alignments(Path(args['DATA_PATH']))
