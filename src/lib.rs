//! # masker
//!
//! This crate provides an object [`Masker`] which can replace
//! a set of (potentially overlapping) patterns in input data
//! with a fixed mask. It's usually faster than simply doing
//! searches with replacement one by one, and it handles two
//! awkward cases:
//! - It allows you to provide the data in chunks, for example
//!   to mask data as you upload it.
//! - It handles the case where masked regions blend together, for
//!   example if two patterns are present and overlap, then replacing
//!   either of them first leaves characters from the other visible.

#![warn(missing_docs)]
use core::fmt::{Debug, Error, Formatter};
use std::collections::BTreeMap;

/// A pattern to mask
///
/// This consists of an optional fixed prefix, which must match in
/// sequence, followed by an optional suffix drawn from a given
/// alphabet. Either the prefix or suffix can be empty. The prefix, if
/// present, can be masked or preserved.
#[derive(Clone, Eq, PartialEq)]
pub struct MatchData<'a> {
    /// A fixed sequence of bytes to match.
    ///
    /// This may be empty, in which case any suffix bytes anywhere
    /// in the input will be masked.
    pub prefix: &'a [u8],
    /// A set of bytes to match after the prefix.
    ///
    /// This is not a sequence, we match a sequence of bytes drawn
    /// from suffix in any order. This may be empty.
    pub suffix: &'a [u8],
    /// If true, the prefix itself is masked, if false the prefix is
    /// preserved.
    pub mask_prefix: bool,
}

impl<'a> Debug for MatchData<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(
            f,
            "[PDATA {:?} {:?}{}]",
            String::from_utf8_lossy(self.prefix),
            String::from_utf8_lossy(self.suffix),
            if self.mask_prefix { " MP" } else { "" },
        )
    }
}

#[derive(Clone, Eq, PartialEq)]
struct Match<'a, 'b>
where
    'b: 'a,
{
    data: &'a MatchData<'b>,
    match_idx: usize,
    offset: usize,
}

impl<'a, 'b> Match<'a, 'b> {
    pub fn new(data: &'a MatchData<'b>, match_idx: usize, offset: usize) -> Self {
        Self {
            data,
            match_idx,
            offset,
        }
    }

    pub fn index(&self) -> usize {
        self.match_idx
    }

    pub fn past_offset(&self, offset: &usize) -> bool {
        self.offset >= *offset
    }

    pub fn allowed_next(&self) -> &'_ [u8] {
        if self.offset < self.data.prefix.len() {
            &self.data.prefix[self.offset..self.offset + 1]
        } else {
            self.data.suffix
        }
    }

    pub fn try_next(&self, action: u8) -> (Option<Self>, Option<(usize, usize)>) {
        if self.offset < self.data.prefix.len() {
            if action == self.data.prefix[self.offset] {
                let offset = self.offset + 1;
                let span = (self.data.mask_prefix && offset == self.data.prefix.len())
                    .then_some((self.data.prefix.len(), 0));
                (Some(Match::new(self.data, self.match_idx, offset)), span)
            } else {
                (None, None)
            }
        } else if self.data.suffix.contains(&action) {
            if !self.data.prefix.is_empty() {
                // distinguish having matched the prefix from having matched the
                // prefix AND having matched something in the suffix
                let offset = std::cmp::min(self.offset + 1, self.data.prefix.len() + 2);
                let span = if self.data.mask_prefix && !self.data.prefix.is_empty() {
                    Some((2, 0))
                } else {
                    Some((offset - self.data.prefix.len(), 0))
                };
                (Some(Match::new(self.data, self.match_idx, offset)), span)
            } else {
                // We're matching bare characters, so there's no
                // prefix check, and no interesting state to keep
                (None, Some((1, 0)))
            }
        } else {
            (None, None)
        }
    }

    /// How many characters back could potentially mask
    pub fn prefix_length(&self) -> usize {
        let pfx = if self.data.mask_prefix {
            self.offset
        } else {
            0
        };

        if self.offset < self.data.prefix.len() + 1 {
            pfx
        } else {
            pfx + (self.offset - self.data.prefix.len())
        }
    }
}

impl<'a, 'b> Debug for Match<'a, 'b> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(
            f,
            "[PREFIX '{}' '{}' {}]",
            String::from_utf8_lossy(self.data.prefix),
            String::from_utf8_lossy(self.data.suffix),
            self.offset
        )
    }
}

#[derive(Clone, Default, PartialEq, Eq)]
struct State<'a, 'b> {
    matches: Vec<Match<'a, 'b>>,
    text: Vec<u8>,
    spans: Vec<(usize, usize)>,
    text_offset: usize,
}

impl<'a, 'b> State<'a, 'b> {
    fn new(
        matches: Vec<Match<'a, 'b>>,
        text: Vec<u8>,
        spans: Vec<(usize, usize)>,
        text_offset: usize,
    ) -> Self {
        Self {
            matches,
            text,
            spans,
            text_offset,
        }
    }

    fn generate_actions(&self, datas: &[MatchData]) -> Vec<Option<u8>> {
        let mut res = Vec::new();
        for pfx in self.matches.iter() {
            for ch in pfx.allowed_next() {
                if !res.contains(&Some(*ch)) {
                    res.push(Some(*ch));
                }
            }
        }

        for data in datas.iter() {
            if !data.prefix.is_empty() {
                let ch = data.prefix[0];
                if !res.contains(&Some(ch)) {
                    res.push(Some(ch))
                }
            } else {
                for ch in data.suffix.as_ref().iter() {
                    if !res.contains(&Some(*ch)) {
                        res.push(Some(*ch))
                    }
                }
            }
        }

        res.push(None);
        res
    }
}

impl<'a, 'b> std::fmt::Debug for State<'a, 'b> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(
            f,
            "[STATE '{}' {:?}",
            String::from_utf8_lossy(&self.text),
            self.spans
        )?;
        for s in self.matches.iter() {
            write!(f, " {:?}", s)?;
        }
        write!(f, " TxtOff: {}", self.text_offset)?;
        write!(f, "]")
    }
}

#[derive(Clone, Default, PartialEq, Eq, Ord, PartialOrd)]
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

impl Debug for Link {
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

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
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
    if spans.is_empty() {
        return Vec::new();
    }

    let mut buf = spans.to_vec();
    buf.sort();
    let mut res = Vec::new();

    let mut cur_span = buf.first().copied().unwrap();
    for span in buf.iter() {
        let new_cur_span = if span.0 < cur_span.1 {
            (cur_span.0, std::cmp::max(cur_span.1, span.1))
        } else {
            res.push(cur_span);
            *span
        };
        cur_span = new_cur_span;
    }
    res.push(cur_span);

    res
}

fn mask_spans<'a>(
    spans: &[(usize, usize)],
    input: &'a [u8],
    mask: &[u8],
    offset: usize,
) -> Vec<u8> {
    let mut res = Vec::new();
    let mut span = 0;

    while span < spans.len() {
        if spans[span].0 >= offset {
            break;
        }
        res.extend_from_slice(mask);
        if spans[span].1 > offset {
            break;
        }
        span += 1;
    }

    for (i, ch) in input.iter().enumerate().map(|(i, ch)| (i + offset, ch)) {
        if span == spans.len() || i < spans[span].0 {
            res.push(*ch);
        } else {
            if i == spans[span].0 {
                res.extend_from_slice(mask);
            }
            if i + 1 == spans[span].1 {
                span += 1;
            }
        }
    }
    res
}

/// Replace byte sequences with a chosen mask value
///
/// This is the central object of this crate. When creating one, an
/// FSM is constructed that can run over a block of data, byte by
/// byte, and replace a set of pre-selected patterns by a mask
/// pattern.
///
/// You can provide all the data up front, using
/// [`mask_slice`](Masker::mask_slice) and
/// [`mask_str`](Masker::mask_str), or you can opt to stream your data
/// through it using [`mask_chunks`](Masker::mask_chunks).
///
/// Example:
/// ```rust
/// use masker::Masker;
///
/// let m = Masker::new(&["frog", "cat"], "XXXX");
/// let s = m.mask_str("the bad frog sat on the cat");
/// assert_eq!(s.as_str(), "the bad XXXX sat on the XXXX");
/// ```
#[derive(Clone, Debug)]
pub struct Masker {
    links: Links,
    default_links: DefaultLinks,
}

impl Masker {
    /// Create a new masker
    ///
    /// `input_data` are the things you want to mask in any data
    /// that is given to the masker. `mask` is the replacement you
    /// want instead of those input data.
    ///
    /// This builds a finite state machine capable of processing
    /// the given input data and mask, and has non-negligible cost,
    /// similiar to a regex compilation.
    ///
    /// Note that it is permissible for the input data to have overlaps,
    /// e.g. the case where you mask "cater" and "bobcat" is handled.
    ///
    /// Example:
    /// ```rust
    /// use masker::Masker;
    ///
    /// let m = Masker::new(&["cater", "bobcat"], "XXXX");
    /// let s = m.mask_str("what does bobcater do?");
    /// assert_eq!(s.as_str(), "what does XXXX do?");
    /// ```
    pub fn new<S, T>(input_data: &[S], mask: T) -> Masker
    where
        S: AsRef<[u8]>,
        T: AsRef<[u8]>,
    {
        Self::new_with_match_data(input_data, &[], mask)
    }

    /// Create a new masker with prefix support
    ///
    /// `input_data` are the things you want to mask in any data
    /// that is given to the masker. `mask` is the replacement you
    /// want instead of those input data.
    ///
    /// `match_data` is a slice of [`MatchData`] objects that describe
    /// some prefix/suffix pairs you would like to mask. This is
    /// strictly a superset of the capabilities provided in `input_data`,
    /// it's just less convenient to construct. Any entry in `input_data`
    /// is equivalent to a [`MatchData`] with the prefix set to the value,
    /// suffix empty, amd `mask_prefix` set to `true`.
    ///
    /// This builds a finite state machine capable of processing
    /// the given inputs and mask, and has non-negligible cost,
    /// similiar to a regex compilation.
    ///
    /// Note that it is permissible for the input data to have
    /// overlaps, e.g. the case where you mask "cater" and "bobcat" is
    /// handled. This extends to the case of token prefixes.
    ///
    /// Example:
    /// ```rust
    /// use masker::{Masker, MatchData};
    ///
    /// let input_data: &[&str] = &[];
    /// let match_data = &[ MatchData { prefix: "secret".as_ref(), suffix: "abc".as_ref(), mask_prefix: true } ];
    /// let m = Masker::new_with_match_data(input_data, match_data, "XXXX".as_bytes());
    /// let s = m.mask_str("what does secretbaaa do?");
    /// assert_eq!(s.as_str(), "what does XXXX do?");
    /// ```
    pub fn new_with_match_data<S, T>(input_data: &[S], match_data: &[MatchData], mask: T) -> Masker
    where
        S: AsRef<[u8]>,
        T: AsRef<[u8]>,
    {
        let prefix_data = input_data
            .iter()
            .map(|s| MatchData {
                prefix: s.as_ref(),
                suffix: &[],
                mask_prefix: true,
            })
            .chain(match_data.iter().cloned())
            .collect::<Vec<_>>();

        let mut states: Vec<State<'_, '_>> = vec![Default::default()];
        let mut links = Vec::new();
        let mut default_links = BTreeMap::new();
        let mut work = vec![0usize];

        let mut coverage = BTreeMap::new();
        for d1 in 0..prefix_data.len() {
            for d2 in 0..prefix_data.len() {
                let mut failed = false;
                // First try to match d2's prefix, starting at this point
                for j in 0..prefix_data[d2].prefix.len() {
                    if !prefix_data[d1].suffix.contains(&prefix_data[d2].prefix[j]) {
                        failed = true;
                        break;
                    }
                }
                if failed {
                    continue;
                }
                for ch in prefix_data[d2].suffix {
                    if !prefix_data[d1].suffix.contains(ch) {
                        failed = true;
                        break;
                    }
                }
                if failed {
                    continue;
                }
                // So now they cover
                coverage.insert((d1, d2), prefix_data[d1].prefix.len() + 1);
            }
        }

        while let Some(index) = work.pop() {
            let actions = states[index].generate_actions(&prefix_data);

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
                let text_offset = states[index].text_offset;
                let full_text_len = new_text.len() + text_offset;

                if let Some(action) = action {
                    for pfx in states[index].matches.iter() {
                        let (pfx, span) = pfx.try_next(action);
                        if let Some(new_pfx) = pfx {
                            if !new_matches.contains(&new_pfx) {
                                new_matches.push(new_pfx);
                            }
                        }
                        if let Some((s1, s2)) = span {
                            new_spans.push((
                                full_text_len - std::cmp::min(full_text_len, s1),
                                full_text_len - std::cmp::min(full_text_len, s2),
                            ));
                        }
                    }

                    for (ix, data) in prefix_data.iter().enumerate() {
                        let mut covered = false;
                        for pfx in states[index].matches.iter() {
                            if let Some(start) = coverage.get(&(pfx.index(), ix)) {
                                if pfx.past_offset(start) {
                                    covered = true;
                                    break;
                                }
                            }
                        }
                        if covered {
                            continue;
                        }

                        let pfx = Match::new(data, ix, 0);
                        let (pfx, span) = pfx.try_next(action);
                        if let Some(new_pfx) = pfx {
                            if !new_matches.contains(&new_pfx) {
                                new_matches.push(new_pfx);
                            }
                        }
                        if let Some((s1, s2)) = span {
                            new_spans.push((
                                full_text_len - std::cmp::min(full_text_len, s1),
                                full_text_len - std::cmp::min(full_text_len, s2),
                            ));
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
                let mut first_kept_char = full_text_len - std::cmp::min(full_text_len, new_extent);

                for (x1, x2) in unified_spans {
                    // This span does not overlap with any span we are
                    // building
                    if x2 + new_extent <= full_text_len {
                        emitted_spans.push((x1, x2));
                    } else {
                        kept_spans.push((x1, x2));
                        // This span overlaps some new span, which means we
                        // may need to keep this text too
                        first_kept_char = std::cmp::min(first_kept_char, x1);
                    }
                }

                let emitted_text = if first_kept_char > 0 {
                    let s = mask_spans(
                        &emitted_spans,
                        &new_text[0..(first_kept_char - text_offset)],
                        mask.as_ref(),
                        text_offset,
                    );
                    if !s.is_empty() {
                        Some(s)
                    } else {
                        None
                    }
                } else {
                    None
                };

                // Prune all emitted text
                let (new_text, new_text_offset) = if first_kept_char > text_offset {
                    (&new_text[(first_kept_char - text_offset)..], 0)
                } else {
                    (new_text.as_slice(), text_offset - first_kept_char)
                };

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
                        first_span.1
                    } else {
                        0
                    }
                } else {
                    0
                };

                let (new_text, new_text_offset) = if cleared > 0 {
                    if cleared > new_text_offset {
                        (&new_text[(cleared - new_text_offset)..], 1)
                    } else {
                        (new_text, new_text_offset - cleared + 1)
                    }
                } else {
                    (new_text, 0)
                };
                let kept_spans = if cleared > 0 {
                    kept_spans
                        .into_iter()
                        .map(|(a, b)| {
                            (
                                a - std::cmp::min(a, cleared - 1),
                                b - std::cmp::min(b, cleared - 1),
                            )
                        })
                        .collect::<Vec<_>>()
                } else {
                    kept_spans
                };

                let new_state =
                    State::new(new_matches, new_text.to_vec(), kept_spans, new_text_offset);

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

    /// Apply masking to a slice of data
    ///
    /// All patterns and overlaps found within the given data block
    /// will be replaced with the previously chosen mask value.
    ///
    /// Example
    /// ```rust
    /// use masker::Masker;
    ///
    /// let m = Masker::new(&["frog", "cat"], "XXXX");
    /// let v = m.mask_slice("the bad frog sat on the cat".as_bytes());
    /// assert_eq!(v.as_slice(), "the bad XXXX sat on the XXXX".as_bytes());
    /// ```
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

    /// Apply masking to text
    ///
    /// All patterns and overlaps found within the given data block
    /// will be replace with the previously chosen mask value.
    ///
    /// This is simply a convenience wrapper over
    /// [`mask_slice`](Masker::mask_slice).
    pub fn mask_str<S>(&self, input: S) -> String
    where
        S: AsRef<str>,
    {
        String::from_utf8(self.mask_slice(input.as_ref())).unwrap()
    }

    /// Begin masking a stream of data chunks
    ///
    /// The return value is an object that will handle the sequence.
    /// It has to be this way because we need to cope with masking
    /// over chunk boundaries, and for that reason there can be some
    /// limited buffering introduced here.
    ///
    /// Example:
    /// ```rust
    /// use masker::Masker;
    ///
    /// let m = Masker::new(&["frog", "cat"], "XXXX");
    /// let mut cm = m.mask_chunks();
    /// let mut v = Vec::new();
    /// v.extend(cm.mask_chunk("the ba"));
    /// v.extend(cm.mask_chunk("d f"));
    /// v.extend(cm.mask_chunk("rog sat on the c"));
    /// v.extend(cm.mask_chunk("at"));
    /// v.extend(cm.finish());
    ///
    /// assert_eq!(v.as_slice(), "the bad XXXX sat on the XXXX".as_bytes());
    /// ```
    pub fn mask_chunks(&self) -> ChunkMasker<'_> {
        ChunkMasker::new(self)
    }
}

/// Mask a sequence of data blocks
///
/// This is the return value from [`Masker::mask_chunks`] and cannot
/// be constructed directly. It expects data to be fed into it
/// sequentially and will correctly mask patterns that cross chunk
/// boundaries.
pub struct ChunkMasker<'a> {
    owner: &'a Masker,
    state: usize,
}

impl<'a> ChunkMasker<'a> {
    fn new(owner: &'a Masker) -> Self {
        Self { owner, state: 0 }
    }

    /// Process the next block of data
    ///
    /// Not all the output from this block will necessarily be
    /// produced as a result of this call; some buffering will be
    /// required if there is a possible match that extends past the
    /// end of the chunk.  Similarly, some output may be produced from
    /// previous chunks, if buffering was introduced there.
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

    /// Finish streaming and flush buffers
    ///
    /// This indicates that there is no further data to come, so any
    /// potential partial matches that have led to buffering can be
    /// resolved, and output produced for them.
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
    use std::time::Instant;

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
                        for b in buf1.iter().take(i) {
                            buf2.push(*b);
                        }
                        buf2.push((
                            std::cmp::min(buf1[i].0, buf1[j].0),
                            std::cmp::max(buf1[i].1, buf1[j].1),
                        ));
                        for b in buf1.iter().take(j).skip(i + 1) {
                            buf2.push(*b);
                        }
                        for b in buf1.iter().skip(j + 1) {
                            buf2.push(*b);
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

    #[test]
    fn test_union() {
        let mut rng = StdRng::seed_from_u64(0xdeadbeefabadcafe);
        for _ in 0..2000000 {
            let mut spans = Vec::new();
            let count = rng.gen_range(1..20);
            for _ in 0..count {
                let x1 = rng.gen_range(0..50);
                let x2 = x1 + rng.gen_range(1..20);
                spans.push((x1, x2));
            }
            let mut value = super::unify_spans(&spans);
            let mut check = slow_union(&spans);
            value.sort();
            check.sort();
            assert_eq!(value, check);
        }
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

        let mut unioned_spans = super::unify_spans(&spans);
        unioned_spans.sort();

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

    fn random_buffer<R: Rng>(mut rng: R, len: usize) -> Vec<u8> {
        let mut res = Vec::new();
        res.resize(len, 0);
        rng.fill_bytes(res.as_mut());
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
        let m = Masker::new(&["abcd", "1ab", "cde", "bce", "aa"], "-MASKED-");
        assert_eq!(m.mask_str("1abcdef"), "-MASKED-f".to_string());
        assert_eq!(m.mask_str("1a"), "1a".to_string());
        assert_eq!(m.mask_str("qqcdeblah"), "qq-MASKED-blah");
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

            let m = Masker::new(&keys, "X");

            for _ in 0..1000 {
                let len = rng.gen_range(0..100);
                let input = random_input(&mut rng, &keys, len);
                let output_as_string = m.mask_str(&input);
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

    fn slice_contains_slice(haystack: &[u8], needle: &[u8]) -> bool {
        haystack
            .windows(needle.len())
            .any(|window| window == needle)
    }

    fn add_separate_keys<R: Rng, S: AsRef<[u8]>>(
        mut rng: R,
        keys: &[S],
        buf: &mut Vec<u8>,
        gap: usize,
    ) -> usize {
        let mut offset = 0;
        let mut keys_added = 0;
        loop {
            let step = rng.gen_range((gap / 2)..gap);
            let key = &keys[rng.gen_range(0..keys.len())];
            offset += step;
            if offset >= buf.len() {
                break;
            }
            let end = std::cmp::min(buf.len(), offset + key.as_ref().len());
            let len = end - offset;
            buf[offset..(len + offset)].copy_from_slice(&key.as_ref()[..len]);
            keys_added += 1;
            offset += len;
        }
        keys_added
    }

    fn add_random_keys<R: Rng, S: AsRef<[u8]>>(
        mut rng: R,
        keys: &[S],
        buf: &mut Vec<u8>,
        count: usize,
    ) -> usize {
        for _ in 0..count {
            let key = &keys[rng.gen_range(0..keys.len())];
            let offset = rng.gen_range(0..buf.len());
            let end = std::cmp::min(buf.len(), offset + key.as_ref().len());
            let len = end - offset;
            buf[offset..(len + offset)].copy_from_slice(&key.as_ref()[..len]);
        }
        count
    }

    fn diff_buffers<A: AsRef<[u8]>, B: AsRef<[u8]>>(a: A, b: B) -> bool {
        let len = std::cmp::min(a.as_ref().len(), b.as_ref().len());
        let mut offset = None;
        for i in 0..len {
            if a.as_ref()[i] != b.as_ref()[i] {
                offset = Some(i);
                break;
            }
        }
        if let Some(offset) = offset {
            println!("A   B   {}", offset);
            let start = if offset < 16 { offset } else { offset - 16 };
            let end = if offset + 16 > len { len } else { offset + 16 };
            for i in start..end {
                println!("{:03} {:03}", a.as_ref()[i], b.as_ref()[i]);
            }
            return false;
        } else if a.as_ref().len() > b.as_ref().len() {
            println!("A   B   {}", b.as_ref().len());
            for i in b.as_ref().len()..a.as_ref().len() {
                println!("{:03} ---", a.as_ref()[i]);
            }
            return false;
        } else if a.as_ref().len() < b.as_ref().len() {
            println!("A   B   {}", a.as_ref().len());
            for i in a.as_ref().len()..b.as_ref().len() {
                println!("--- {:03}", b.as_ref()[i]);
            }
            return false;
        }
        true
    }

    fn mask_slice_check<S, T, U>(input: S, mask: T, keys: &[U]) -> Vec<u8>
    where
        S: AsRef<[u8]>,
        T: AsRef<[u8]>,
        U: AsRef<[u8]>,
    {
        let spans = {
            let mut spans = Vec::new();
            for key in keys.iter() {
                for ix in input
                    .as_ref()
                    .windows(key.as_ref().len())
                    .enumerate()
                    .filter(|(_, window)| window == &key.as_ref())
                    .map(|(index, _)| index)
                {
                    let len = key.as_ref().len();
                    spans.push((ix, ix + len));
                }
            }
            spans
        };

        let mut unioned_spans = super::unify_spans(&spans);
        unioned_spans.sort();

        let mut offset = 0usize;
        let mut res = Vec::new();
        for span in unioned_spans {
            if offset < span.0 {
                res.extend_from_slice(&input.as_ref()[offset..span.0]);
            }
            res.extend_from_slice(mask.as_ref());
            offset = span.1;
        }
        if offset < input.as_ref().len() {
            res.extend_from_slice(&input.as_ref()[offset..]);
        }
        res
    }

    #[test]
    fn test_masker_slabs() {
        let mut rng = StdRng::seed_from_u64(0xdeadbeefabadcafe);
        for input_type in 0..4 {
            for _ in 0..2 {
                let num_keys = rng.gen_range(1..15);
                let mut keys = Vec::new();
                for _ in 0..num_keys {
                    let len = rng.gen_range(10..50);
                    keys.push(random_buffer(&mut rng, len));
                }

                let m = Masker::new(&keys, "XXXX-XXXX-XXXX-XXXX");

                for _ in 0..3 {
                    let len = rng.gen_range(5_000_000..100_000_000);
                    let mut input = random_buffer(&mut rng, len);
                    let key_count = match input_type {
                        0 => 0,
                        1 => add_random_keys(&mut rng, &keys, &mut input, 5),
                        2 => add_random_keys(&mut rng, &keys, &mut input, 20),
                        3 => add_separate_keys(&mut rng, &keys, &mut input, 20000),
                        _ => unreachable!(),
                    };
                    let output_start = Instant::now();
                    let output = m.mask_slice(&input);
                    let output_time = Instant::now().duration_since(output_start);
                    let check_start = Instant::now();
                    let check = mask_slice_check(&input, "XXXX-XXXX-XXXX-XXXX", &keys);
                    let check_time = Instant::now().duration_since(check_start);
                    println!(
                        "Buffer {} Keys: {} Mask time: {} Check time: {}",
                        len,
                        key_count,
                        output_time.as_secs_f64(),
                        check_time.as_secs_f64()
                    );
                    for key in keys.iter() {
                        assert!(
                            !slice_contains_slice(&output, key),
                            "Key {:?} is contained in output",
                            key
                        );
                    }
                    for key in keys.iter() {
                        assert!(
                            !slice_contains_slice(&check, key),
                            "Key {:?} is contained in check",
                            key
                        );
                    }
                    diff_buffers(&output, &check);
                    assert_eq!(output, check);
                }
            }
        }
    }

    #[test]
    fn test_chunk_masker() {
        let m = Masker::new(&["abcd", "1ab", "cde", "bce", "aa"], "-MASK-");
        let mut cm = m.mask_chunks();
        assert_eq!(cm.mask_chunk("ab"), Vec::new());
        assert_eq!(cm.mask_chunk("c"), Vec::new());
        assert_eq!(cm.mask_chunk("d"), Vec::new());
        assert_eq!(cm.mask_chunk("g"), Vec::from("-MASK-g".as_bytes()));
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

            let m = Masker::new(&keys, "X");

            for _ in 0..1000 {
                let len = rng.gen_range(0..100);
                let input = random_input(&mut rng, &keys, len);
                let mut cm = m.mask_chunks();
                let mut output = Vec::new();
                let mut offset = 0;
                while offset < input.len() {
                    let chunk_len = rng.gen_range(0..(std::cmp::min(10, input.len() - offset + 1)));
                    let mut chunk = Vec::new();
                    for _ in 0..chunk_len {
                        chunk.push(input.as_bytes()[offset]);
                        offset += 1;
                    }
                    output.extend_from_slice(cm.mask_chunk(chunk).as_ref());
                }
                output.extend(cm.finish().as_slice());
                let output_as_string = String::from_utf8_lossy(&output);
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
    fn test_chunk_masker_slabs() {
        let mut rng = StdRng::seed_from_u64(0xdeadbeefabadcafe);
        for input_type in 0..4 {
            for _ in 0..2 {
                let num_keys = rng.gen_range(1..15);
                let mut keys = Vec::new();
                for _ in 0..num_keys {
                    let len = rng.gen_range(10..50);
                    keys.push(random_buffer(&mut rng, len));
                }

                let m = Masker::new(&keys, "XXXX-XXXX-XXXX-XXXX");

                for _ in 0..3 {
                    let len = rng.gen_range(5_000_000..100_000_000);
                    let mut input = random_buffer(&mut rng, len);
                    let key_count = match input_type {
                        0 => 0,
                        1 => add_random_keys(&mut rng, &keys, &mut input, 5),
                        2 => add_random_keys(&mut rng, &keys, &mut input, 20),
                        3 => add_separate_keys(&mut rng, &keys, &mut input, 20000),
                        _ => unreachable!(),
                    };
                    let output_start = Instant::now();
                    let mut cm = m.mask_chunks();
                    let mut output = Vec::new();
                    let mut offset = 0;
                    while offset < input.len() {
                        let chunk_len =
                            rng.gen_range(0..(std::cmp::min(10, input.len() - offset + 1)));
                        let mut chunk = Vec::new();
                        for _ in 0..chunk_len {
                            chunk.push(input[offset]);
                            offset += 1;
                        }
                        output.extend_from_slice(cm.mask_chunk(chunk).as_ref());
                    }
                    output.extend(cm.finish().as_slice());
                    let output_time = Instant::now().duration_since(output_start);

                    let check_start = Instant::now();
                    let check = mask_slice_check(&input, "XXXX-XXXX-XXXX-XXXX", &keys);
                    let check_time = Instant::now().duration_since(check_start);
                    println!(
                        "Buffer {} Keys: {} Mask time: {} Check time: {}",
                        len,
                        key_count,
                        output_time.as_secs_f64(),
                        check_time.as_secs_f64()
                    );
                    for key in keys.iter() {
                        assert!(
                            !slice_contains_slice(&output, key),
                            "Key {:?} is contained in output {:?}",
                            key,
                            output
                        );
                    }
                    diff_buffers(&output, &check);
                    assert_eq!(output, check);
                }
            }
        }
    }
}
