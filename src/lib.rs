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

impl<'a> Display for Match<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(
            f,
            "[MATCH '{}' {}]",
            String::from_utf8_lossy(self.input),
            self.offset
        )
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

impl<'a> std::fmt::Display for State<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "[STATE")?;
        for s in self.matches.iter() {
            write!(f, " {}", s)?;
        }
        write!(f, "]")
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

impl Display for Link {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(
            f,
            r#"[LINK {} -> {} "{}" ({})"#,
            self.source,
            self.target,
            char::from_u32(self.action as u32).unwrap_or('?'),
            self.action
        )
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
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn slow_union(input: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let mut buf1 = Vec::from(input);
        let mut buf2 = Vec::new();
        let mut changes = true;
        while changes {
            changes = false;
            for i in 0..buf1.len() {
                for j in (i + 1)..buf1.len() {
                    let x1 = std::cmp::max(buf1[i].0, buf1[j].0);
                    let x2 = std::cmp::min(buf1[i].1, buf1[j].1);
                    if x1 < x2 {
                        // overlap
                        for x in 0..i {
                            buf2.push(buf1[x]);
                        }
                        buf2.push((
                            std::cmp::min(buf1[i].0, buf1[j].0),
                            std::cmp::max(buf1[i].1, buf1[j].1),
                        ));
                        for x in i + 1..j {
                            buf2.push(buf1[x]);
                        }
                        for x in j + 1..buf1.len() {
                            buf2.push(buf1[x]);
                        }
                        std::mem::swap(&mut buf1, &mut buf2);
                        buf2.clear();
                        changes = true;
                        break;
                    }
                }
                if changes {
                    break;
                }
            }
        }
        buf1.sort_by(|a, b| a.0.cmp(&b.0));
        buf1
    }

    fn mask_string_check<S: AsRef<str>>(string: &str, mask: &str, keys: &[S]) -> String {
        let spans = {
            let mut spans = Vec::new();
            for key in keys.iter() {
                let mut offset = 0usize;
                while let Some(ix) = string[offset..].find(key.as_ref()) {
                    let len = key.as_ref().as_bytes().len();
                    spans.push((offset + ix, offset + ix + len));
                    offset += ix + 1;
                }
            }
            spans
        };

        let unioned_spans = slow_union(&spans);

        let mut offset = 0usize;
        let mut res = Vec::new();
        for span in unioned_spans {
            if offset < span.0 {
                res.extend_from_slice(&string.as_bytes()[offset..span.0]);
            }
            res.extend_from_slice(mask.as_bytes());
            offset = span.1;
        }
        if offset < string.as_bytes().len() {
            res.extend_from_slice(&string.as_bytes()[offset..]);
        }
        String::from_utf8_lossy(&res).into()
    }

    fn random_string<R: Rng>(mut rng: R, len: usize) -> String {
        let mut res = String::new();
        for _ in 0..len {
            let ch = rng.gen_range('a'..'e');
            res.push(ch);
        }
        res
    }

    fn random_input<R: Rng>(mut rng: R, keys: &Vec<String>, len: usize) -> String {
        let mut res = String::new();
        // Because we require the chunks not contain any keys, and we
        // have a limited alphabet, making max_chunk large makes the
        // tests run very slowly, and in some cases can essentially
        // stall.
        let max_chunk = std::cmp::min(5, (len / 4) + 1);
        let mut stage = 0;
        assert!(max_chunk > 0);
        while res.len() < len {
            if stage == 0 {
                let len = rng.gen_range(1..(max_chunk + 1));
                if len > 0 {
                    let mut remaining = 1000;
                    let chunk = loop {
                        let chunk = random_string(&mut rng, len);
                        if !keys.iter().any(|k| chunk.contains(k)) {
                            break chunk;
                        }
                        remaining -= 1;
                        if remaining == 0 {
                            break String::new();
                        }
                    };
                    res.push_str(&chunk);
                }
            } else if !keys.is_empty() {
                let key = rng.gen_range(0..keys.len());
                res.push_str(&keys[key]);
            }
            stage = 1 - stage;
        }
        res
    }

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

    #[test]
    fn test_masker_random() {
        let mut rng = StdRng::seed_from_u64(0xdeadbeefabadcafe);
        for _ in 0..2000 {
            let num_keys = rng.gen_range(0..5);
            let mut keys = Vec::new();
            for _ in 0..num_keys {
                let len = rng.gen_range(1..6);
                keys.push(random_string(&mut rng, len));
            }

            let m = Masker::new(&keys);

            for _ in 0..100 {
                let len = rng.gen_range(0..100);
                let input = random_input(&mut rng, &keys, len);
                let output_as_string = m.mask_string(&input, "X");
                let check = mask_string_check(&input, "X", &keys);
                for key in keys.iter() {
                    assert!(
                        !output_as_string.contains(key),
                        "Key {} is contained in output {}",
                        key,
                        output_as_string
                    );
                }
                assert_eq!(output_as_string, check);
            }
        }
    }

    #[test]
    fn test_chunk_masker() {
        let m = Masker::new(&vec!["abcd", "1ab", "cde", "bce", "aa"]);
        let mut cm = m.mask_chunks("-MASK-");
        assert_eq!(cm.mask_chunk("ab".as_ref()).to_owned(), "".as_ref());
        assert_eq!(cm.mask_chunk("c".as_ref()).to_owned(), "".as_ref());
        assert_eq!(cm.mask_chunk("d".as_ref()).to_owned(), "".as_ref());
        assert_eq!(cm.mask_chunk("g".as_ref()).to_owned(), "-MASK-g".as_ref());
        assert_eq!(cm.finish().as_slice(), "".as_bytes())
    }

    #[test]
    fn test_chunk_masker_random() {
        let mut rng = StdRng::seed_from_u64(0xdeadbeefabadcafe);
        for _ in 0..2000 {
            let num_keys = rng.gen_range(1..=5);
            let mut keys = Vec::new();
            for _ in 0..num_keys {
                let len = rng.gen_range(1..6);
                keys.push(random_string(&mut rng, len));
            }

            let m = Masker::new(&keys);

            for _ in 0..100 {
                let len = rng.gen_range(0..100);
                let input = random_input(&mut rng, &keys, len);
                let mut cm = m.mask_chunks("X");
                let mut output = Vec::new();
                let mut offset = 0;
                while offset < input.len() {
                    let chunk_len = rng.gen_range(0..(std::cmp::min(10, input.len() - offset + 1)));
                    let mut chunk = Vec::new();
                    for _ in 0..chunk_len {
                        chunk.push(input.as_bytes()[offset]);
                        offset += 1;
                    }
                    output.extend_from_slice(cm.mask_chunk(chunk.as_ref()).as_ref());
                }
                output.extend(cm.finish().as_slice());
                let output_as_string = String::from_utf8_lossy(&output);
                let check = mask_string_check(&input, "X", &keys);
                println!("Took input      {}", input);
                println!("Check           {}", check);
                println!("Keys            {:?}", keys);
                println!("Produced output {}", String::from_utf8_lossy(&output));
                for key in keys.iter() {
                    assert!(
                        !output_as_string.contains(key),
                        "Key {} is contained in output {}",
                        key,
                        output_as_string
                    );
                }
                assert_eq!(output_as_string, check);
            }
        }
    }
}
