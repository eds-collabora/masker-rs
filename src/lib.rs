use core::cmp::max;
use core::fmt::{Display, Error, Formatter};
use std::borrow::Cow;
use std::collections::{BTreeSet, VecDeque};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Match<'a> {
    input: &'a [u8],
    offset: usize,
}

impl<'a> Match<'a> {
    pub fn new(input: &'a [u8], offset: usize) -> Self {
        Self { input, offset }
    }

    pub fn next(&self) -> u8 {
        self.input[self.offset]
    }

    pub fn len(&self) -> usize {
        self.input.len()
    }

    pub fn prefix_length(&self) -> usize {
        self.offset
    }

    pub fn try_next(&self, action: u8) -> Option<Self> {
        if action == self.input[self.offset] && self.offset + 1 < self.input.len() {
            Some(Match::new(self.input, self.offset + 1))
        } else {
            None
        }
    }

    pub fn completed_by(&self, action: u8) -> bool {
        action == self.input[self.offset] && self.offset + 1 == self.input.len()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct State<'a> {
    matches: Vec<Match<'a>>,
}

impl<'a> State<'a> {
    fn new(matches: Vec<Match<'a>>) -> Self {
        Self { matches }
    }

    /// Return the most preceding characters that could be part of a
    /// match if we are in this state.
    fn longest_prefix(&self) -> Option<usize> {
        self.matches.iter().map(Match::prefix_length).max()
    }

    fn generate_actions<S>(&self, strings: &[S]) -> Vec<u8>
    where
        S: AsRef<[u8]>,
    {
        let mut res = Vec::new();
        for mtch in self.matches.iter() {
            let ch = mtch.next();
            if !res.contains(&ch) {
                res.push(ch);
            }
        }
        for input in strings.iter() {
            let ch = input.as_ref()[0];
            if !res.contains(&ch) {
                res.push(ch);
            }
        }
        res
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Ord, PartialOrd)]
struct Link {
    source: usize,
    target: usize,
    action: u8,
    completion: Option<usize>,
}

impl Link {
    pub fn new(source: usize, target: usize, action: u8, completion: Option<usize>) -> Self {
        Self {
            source,
            target,
            action,
            completion,
        }
    }
}

fn unify_spans(spans: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut res = Vec::<(usize, usize)>::new();
    let mut new_res = Vec::<(usize, usize)>::new();
    for span in spans.iter() {
        let (mut x1, mut x2) = *span;
        for old_span in res.iter() {
            if span.0 < old_span.0 {
                if span.1 <= old_span.0 {
                    new_res.push(*old_span);
                } else {
                    x2 = max(x2, old_span.1)
                }
            } else if old_span.1 <= span.0 {
                new_res.push(*old_span);
            } else {
                x1 = old_span.0;
                x2 = max(x2, old_span.1);
            }
        }
        new_res.push((x1, x2));
        std::mem::swap(&mut res, &mut new_res);
        new_res.clear()
    }
    res
}

pub struct Masker {
    prefixes: Vec<usize>,
    links: Vec<Link>,
}

impl Masker {
    pub fn new<S>(input_strings: &[S]) -> Masker
    where
        S: AsRef<[u8]>,
    {
        let mut states = vec![State::new(Default::default())];
        let mut links = BTreeSet::new();
        let mut work = vec![0usize];

        while let Some(index) = work.pop() {
            let actions = states[index].generate_actions(input_strings);

            for action in actions {
                let mut new_matches = Vec::new();
                let mut completed = None;

                for mtch in states[index].matches.iter() {
                    if let Some(new_match) = mtch.try_next(action) {
                        new_matches.push(new_match);
                    } else if mtch.completed_by(action) {
                        completed = Some(if let Some(length) = completed {
                            max(length, mtch.len())
                        } else {
                            mtch.len()
                        });
                    }
                }

                for input in input_strings.iter() {
                    let mtch = Match::new(input.as_ref(), 0);
                    if let Some(new_match) = mtch.try_next(action) {
                        new_matches.push(new_match)
                    } else if mtch.completed_by(action) {
                        completed = Some(if let Some(length) = completed {
                            max(length, mtch.len())
                        } else {
                            mtch.len()
                        });
                    }
                    // if input.as_ref()[0] == action {
                    //     new_matches.push(Match::new(input.as_ref(), 1));
                    // }
                }

                let new_state = State::new(new_matches);
                let new_index = if let Some(new_index) = states.iter().position(|x| x == &new_state)
                {
                    new_index
                } else {
                    let new_index = states.len();
                    states.push(new_state);
                    work.push(new_index);
                    new_index
                };

                links.insert(Link::new(index, new_index, action, completed));
            }
        }

        Self {
            prefixes: states
                .into_iter()
                .map(|s| s.longest_prefix().unwrap_or(0usize))
                .collect(),
            links: links.into_iter().collect(),
        }
    }

    fn generate_spans<S>(&self, input: S) -> Vec<(usize, usize)>
    where
        S: AsRef<[u8]>,
    {
        let mut state = 0usize;
        let mut res = Vec::new();
        for (i, ch) in input.as_ref().iter().enumerate() {
            let mut matched = false;
            for link in self.links.iter() {
                if link.source == state && *ch == link.action {
                    if let Some(span) = link.completion {
                        res.push((i + 1 - span, i + 1));
                    }
                    state = link.target;
                    matched = true;
                    break;
                }
            }
            if !matched {
                state = 0;
            }
        }
        res
    }

    pub fn mask_string<'a>(&self, input: &'a str, mask: &str) -> Cow<'a, str> {
        let spans = self.generate_spans(input);
        if spans.is_empty() {
            return input.into();
        }
        let spans = unify_spans(&spans);
        let mut res = String::new();
        let mut span = 0;
        for (i, ch) in input.chars().enumerate() {
            if span == spans.len() || i < spans[span].0 {
                res.push(ch);
            } else {
                if i == spans[span].0 {
                    res.push_str(mask);
                }
                if i + 1 == spans[span].1 {
                    span += 1;
                }
            }
        }
        res.into()
    }

    pub fn mask_chunks<'a, 'b>(&'a self, mask: &'b str) -> ChunkMasker<'a, 'b> {
        ChunkMasker::new(self, mask)
    }
}

pub struct ChunkMasker<'a, 'b> {
    owner: &'a Masker,
    mask: &'b [u8],
    state: usize,
    offset: usize,
    held: VecDeque<u8>,
    held_spans: VecDeque<(usize, usize)>,
}

impl<'a, 'b> ChunkMasker<'a, 'b> {
    fn new(owner: &'a Masker, mask: &'b str) -> Self {
        Self {
            owner,
            mask: mask.as_ref(),
            state: 0,
            offset: 0,
            held: VecDeque::new(),
            held_spans: VecDeque::new(),
        }
    }

    fn generate_spans(&mut self, chunk: &[u8]) -> Vec<(usize, usize)> {
        let mut res = Vec::new();
        let start = self.offset + self.held.len();
        for (i, ch) in chunk.as_ref().iter().enumerate() {
            let mut matched = false;
            for link in self.owner.links.iter() {
                if link.source == self.state && *ch == link.action {
                    if let Some(span) = link.completion {
                        res.push((start + i + 1 - span, start + i + 1));
                    }
                    self.state = link.target;
                    matched = true;
                    break;
                }
            }
            if !matched {
                self.state = 0;
            }
        }
        res
    }

    pub fn mask_chunk<'c>(&mut self, chunk: &'c [u8]) -> Cow<'c, [u8]> {
        let spans = self.generate_spans(chunk.as_ref());
        // (1) We don't have anything queued as possible matches
        // (2) We haven't found anything that needs masking
        // (3) We haven't found anything that might end up needing masking
        if self.held.is_empty() && spans.is_empty() && self.owner.prefixes[self.state] == 0 {
            // FIXME: this is overly conservative, we could return a slice of chunk sometimes
            return chunk.as_ref().into();
        }
        // So now we have the slow path
        self.held.extend(chunk.as_ref().iter());
        self.held_spans.extend(spans);
        // FIXME: we could store the unified spans
        let spans = unify_spans(self.held_spans.make_contiguous());

        //self.held now contains all the unprocessed text
        // emittable_end is the first character that could be part of a new span
        let emittable_end = self.held.len() - self.owner.prefixes[self.state] + self.offset;
        let mut res = Vec::new();
        let mut span = 0;
        for i in self.offset..emittable_end {
            if span == spans.len() || i < spans[span].0 {
                // either there is no span ahead, or we haven't reached it
                let local_ix = i - self.offset;
                res.push(self.held[local_ix]);
            } else {
                if i == spans[span].0 {
                    // we are at the start of a span
                    // check that it doesn't extend past emittable end
                    if spans[span].1 > emittable_end {
                        break;
                    }
                    res.extend_from_slice(self.mask);
                }
                if (i + 1) == spans[span].1 {
                    // we are at the end of a span
                    span += 1;
                }
            }
        }

        let end = self.offset + self.held.len();
        let span_limit = if span < spans.len() {
            spans[span].0
        } else {
            end
        };
        let limit = std::cmp::min(emittable_end, span_limit);

        if span == spans.len() {
            self.held_spans.clear();
        } else {
            while !self.held_spans.is_empty() {
                if self.held_spans[0].1 > spans[span].0 {
                    break;
                }
                self.held_spans.pop_front();
            }
        }
        for _ in self.offset..limit {
            self.held.pop_front();
        }
        self.offset = limit;

        res.into()
    }

    pub fn finish(mut self) -> Vec<u8> {
        if self.held.is_empty() {
            return Vec::new();
        }
        // So now we have the slow path
        let spans = unify_spans(self.held_spans.make_contiguous());
        let mut res = Vec::new();
        let mut span = 0;
        let end = self.held.len() + self.offset;
        for i in self.offset..end {
            if span == spans.len() || i < spans[span].0 {
                let local_ix = i - self.offset;
                res.push(self.held[local_ix]);
            } else {
                if i == spans[span].0 {
                    res.extend_from_slice(self.mask);
                }
                if (i + 1) == spans[span].1 {
                    span += 1;
                }
            }
        }

        res
    }
}

#[cfg(test)]
mod test {
    use super::Masker;

    #[test]
    fn test_masker() {
        let m = Masker::new(&vec!["abcd", "1ab", "cde", "bce", "aa"]);
        assert_eq!(
            m.mask_string("1abcdef", "-MASKED-"),
            "-MASKED-f".to_string()
        );
        assert_eq!(m.mask_string("1a", "-MASKED-"), "1a".to_string());
        assert_eq!(m.mask_string("qqcdeblah", "-MASKED-"), "qq-MASKED-blah");
    }
}
