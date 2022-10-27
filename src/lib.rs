use core::cmp::max;
use core::fmt::{Debug, Display, Error, Formatter};
use std::collections::BTreeMap;

#[derive(Clone, Default, Eq, PartialEq)]
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
    text: Vec<u8>,
    spans: Vec<(usize, usize)>,
}

impl<'a> State<'a> {
    fn new(matches: Vec<Match<'a>>, text: Vec<u8>, spans: Vec<(usize, usize)>) -> Self {
        Self {
            matches,
            text,
            spans,
        }
    }

    fn generate_actions<S>(&self, strings: &[S]) -> Vec<Option<u8>>
    where
        S: AsRef<[u8]>,
    {
        let mut res = Vec::new();
        for mtch in self.matches.iter() {
            let ch = mtch.next();
            if !res.contains(&Some(ch)) {
                res.push(Some(ch));
            }
        }
        for input in strings.iter() {
            let ch = input.as_ref()[0];
            if !res.contains(&Some(ch)) {
                res.push(Some(ch));
            }
        }
        res.push(None);
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
    emitted: Option<Vec<u8>>,
}

impl Link {
    pub fn new(source: usize, target: usize, action: u8, emitted: Option<Vec<u8>>) -> Self {
        Self {
            source,
            target,
            action,
            emitted,
        }
    }
}

impl Display for Link {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(
            f,
            "[LINK {} -> {} '{}' ({}){}",
            self.source,
            self.target,
            char::from_u32(self.action as u32).unwrap_or('?'),
            self.action,
            if let Some(emitted) = &self.emitted {
                format!(r#" "{}""#, String::from_utf8_lossy(emitted))
            } else {
                String::new()
            }
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct LinkKey(usize);

struct Links {
    source_offset: Vec<usize>,
    target: Vec<usize>,
    actions: Vec<u8>,
    emitted: Vec<Option<Vec<u8>>>,
}

impl Links {
    pub fn new(mut links: Vec<Link>) -> Self {
        links.sort_by(|a, b| a.source.cmp(&b.source));
        let mut source_offset = Vec::new();
        let mut target = Vec::new();
        let mut actions = Vec::new();
        let mut emitted = Vec::new();
        let mut prev_state = 0;
        source_offset.push(0);
        for link in links {
            while prev_state < link.source {
                source_offset.push(target.len());
                prev_state += 1;
            }
            target.push(link.target);
            actions.push(link.action);
            emitted.push(link.emitted);
        }
        source_offset.push(target.len());
        Self {
            source_offset,
            target,
            actions,
            emitted,
        }
    }

    pub fn get(&self, state: usize, action: u8) -> Option<LinkKey> {
        let start = self.source_offset[state];
        let end = self.source_offset[state + 1];
        for i in start..end {
            if self.actions[i] == action {
                return Some(LinkKey(i));
            }
        }
        None
    }

    pub fn target(&self, key: LinkKey) -> usize {
        self.target[key.0]
    }

    pub fn emitted(&self, key: LinkKey) -> Option<&Vec<u8>> {
        self.emitted[key.0].as_ref()
    }
}

struct DefaultLinks {
    emitted: Vec<Option<Vec<u8>>>,
}

impl DefaultLinks {
    pub fn new(default_links: BTreeMap<usize, Option<Vec<u8>>>) -> Self {
        let mut emitted = Vec::new();
        for (k, v) in default_links {
            if k >= emitted.len() {
                emitted.resize(k + 1, None);
            }
            emitted[k] = v;
        }
        Self { emitted }
    }

    pub fn get(&self, state: usize) -> Option<&Vec<u8>> {
        self.emitted[state].as_ref()
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

fn mask_spans<'a>(spans: &[(usize, usize)], input: &'a [u8], mask: &[u8]) -> Vec<u8> {
    let mut res = Vec::new();
    let mut span = 0;
    for (i, ch) in input.iter().enumerate() {
        if span == spans.len() || i < spans[span].0 {
            res.push(*ch);
        } else {
            if i == spans[span].0 {
                res.extend(mask.iter());
            }
            if i + 1 == spans[span].1 {
                span += 1;
            }
        }
    }
    res
}

pub struct Masker {
    links: Links,
    default_links: DefaultLinks,
}

impl Masker {
    pub fn new<S, T>(input_data: &[S], mask: T) -> Masker
    where
        S: AsRef<[u8]>,
        T: AsRef<[u8]>,
    {
        let mut states: Vec<State<'_>> = vec![Default::default()];
        let mut links = Vec::new();
        let mut default_links = BTreeMap::new();
        let mut work = vec![0usize];

        while let Some(index) = work.pop() {
            let actions = states[index].generate_actions(input_data);

            for action in actions {
                // STEP 1: Find the new matches, and spans of completed matches
                let mut new_matches = Vec::new();
                let mut new_spans = states[index].spans.to_vec();
                let new_text = {
                    let mut t = states[index].text.clone();
                    if let Some(action) = action {
                        t.push(action);
                    }
                    t
                };

                if let Some(action) = action {
                    for mtch in states[index].matches.iter() {
                        if let Some(new_match) = mtch.try_next(action) {
                            new_matches.push(new_match);
                        } else if mtch.completed_by(action) {
                            new_spans.push((
                                new_text.len() - std::cmp::min(new_text.len(), mtch.len()),
                                new_text.len(),
                            ))
                        }
                    }

                    for input in input_data.iter() {
                        let mtch = Match::new(input.as_ref(), 0);
                        if let Some(new_match) = mtch.try_next(action) {
                            new_matches.push(new_match)
                        } else if mtch.completed_by(action) {
                            new_spans.push((
                                new_text.len() - std::cmp::min(new_text.len(), mtch.len()),
                                new_text.len(),
                            ))
                        }
                    }
                }

                // STEP 2: Find the emitted text, based on the spans present
                let unified_spans = unify_spans(&new_spans);
                let mut emitted_spans = Vec::new();
                let mut kept_spans = Vec::new();
                // Keep any char that may be in the extent of a new span
                let new_extent = new_matches
                    .iter()
                    .map(|m| m.prefix_length())
                    .max()
                    .unwrap_or(0usize);
                let mut first_kept_char =
                    new_text.len() - std::cmp::min(new_text.len(), new_extent);

                for (x1, x2) in unified_spans {
                    // This span does not overlap with any span we are
                    // building
                    if x2 + new_extent <= new_text.len() {
                        emitted_spans.push((x1, x2));
                    } else {
                        kept_spans.push((x1, x2));
                        // This span overlaps some new span, which means we
                        // may need to keep this text too
                        first_kept_char = std::cmp::min(first_kept_char, x1);
                    }
                }

                let emitted_text = if first_kept_char > 0 {
                    Some(mask_spans(
                        &emitted_spans,
                        &new_text[0..first_kept_char],
                        mask.as_ref(),
                    ))
                } else {
                    None
                };

                // Prune all emitted text
                let new_text = &new_text[first_kept_char..];
                // Rebase spans for pruning
                let mut kept_spans = kept_spans
                    .into_iter()
                    .map(|(a, b)| (a - first_kept_char, b - first_kept_char))
                    .collect::<Vec<_>>();
                kept_spans.sort_by(|a: &(usize, usize), b: &(usize, usize)| a.0.cmp(&b.0));

                // STEP 3: Clear any left-anchored text that will be masked,
                //         replacing it with an empty span to mark the spot
                let cleared = if let Some(first_span) = kept_spans.first().copied() {
                    if first_span.0 == 0 && first_span.1 > 0 {
                        first_span.1 - 1
                    } else {
                        0
                    }
                } else {
                    0
                };

                let new_text = &new_text[cleared..];
                let kept_spans = kept_spans
                    .into_iter()
                    .map(|(a, b)| (a - std::cmp::min(a, cleared), b - std::cmp::min(b, cleared)))
                    .collect::<Vec<_>>();

                let new_state = State::new(new_matches, new_text.to_vec(), kept_spans);

                let new_index = if let Some(new_index) = states.iter().position(|x| x == &new_state)
                {
                    new_index
                } else {
                    let new_index = states.len();
                    states.push(new_state);
                    work.push(new_index);
                    new_index
                };

                if let Some(action) = action {
                    let lnk = Link::new(index, new_index, action, emitted_text);
                    links.push(lnk);
                } else {
                    default_links.insert(index, emitted_text);
                }
            }
        }

        Self {
            links: Links::new(links),
            default_links: DefaultLinks::new(default_links),
        }
    }

    pub fn mask_slice<S>(&self, input: S) -> Vec<u8>
    where
        S: AsRef<[u8]>,
    {
        let mut state = 0usize;
        let mut res = Vec::new();
        res.reserve(input.as_ref().len());
        for ch in input.as_ref().iter() {
            if let Some(link) = self.links.get(state, *ch) {
                if let Some(emitted) = self.links.emitted(link) {
                    res.extend(emitted);
                }
                state = self.links.target(link);
            } else {
                if let Some(emitted) = self.default_links.get(state) {
                    res.extend(emitted);
                }
                res.push(*ch);
                state = 0;
            }
        }
        if let Some(emitted) = self.default_links.get(state) {
            res.extend(emitted);
        }
        res
    }

    pub fn mask_str<S>(&self, input: S) -> String
    where
        S: AsRef<str>,
    {
        String::from_utf8(self.mask_slice(input.as_ref())).unwrap()
    }

    pub fn mask_chunks(&self) -> ChunkMasker<'_> {
        ChunkMasker::new(self)
    }
}

pub struct ChunkMasker<'a> {
    owner: &'a Masker,
    state: usize,
}

impl<'a> ChunkMasker<'a> {
    fn new(owner: &'a Masker) -> Self {
        Self { owner, state: 0 }
    }

    pub fn mask_chunk<C>(&mut self, chunk: C) -> Vec<u8>
    where
        C: AsRef<[u8]>,
    {
        let mut res = Vec::new();
        res.reserve(chunk.as_ref().len());
        for ch in chunk.as_ref().iter() {
            if let Some(link) = self.owner.links.get(self.state, *ch) {
                if let Some(emitted) = self.owner.links.emitted(link) {
                    res.extend(emitted);
                }
                self.state = self.owner.links.target(link);
            } else {
                if let Some(emitted) = self.owner.default_links.get(self.state) {
                    res.extend(emitted);
                }
                res.push(*ch);
                self.state = 0;
            }
        }
        res
    }

    pub fn finish(self) -> Vec<u8> {
        let mut res = Vec::new();
        if let Some(emitted) = self.owner.default_links.get(self.state) {
            res.extend(emitted);
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
