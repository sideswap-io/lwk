use crate::descriptor::{Chain, WolletDescriptor};
use crate::elements::{BlockHash, OutPoint, Script, Transaction, TxOutSecrets, Txid};
use crate::hashes::Hash;
use crate::{BlindingPublicKey, Error};
use elements::bitcoin::bip32::ChildNumber;
use lwk_common::{DynStore, MemoryStore};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

pub const BATCH_SIZE: u32 = 20;
pub type Height = u32;
pub type Timestamp = u32;

const TXIDS_KEY: &str = "wollet:txids";
const CANNOT_UNBLIND_TXIDS_KEY: &str = "wollet:cannot_unblind_txids";

fn tx_key(txid: &Txid) -> String {
    format!("wollet:tx:{}", txid)
}

/// `Cache` is a cache of wallet data, like wallet transactions.
/// It is fully reconstructable from the CT Descriptor and the blockchain.
pub struct Cache {
    /// Store for all wallet transactions
    pub txs_store: Arc<dyn DynStore>,

    /// all txids in txs_store
    pub txids: HashSet<Txid>,

    /// contains all my script up to an empty batch of BATCHSIZE
    pub paths: HashMap<Script, (Chain, ChildNumber)>,

    /// inverse of `paths`, with the blinding public key for each script
    pub scripts: HashMap<(Chain, ChildNumber), (Script, Option<BlindingPublicKey>)>,

    /// contains only my wallet txs with the relative heights (None if unconfirmed)
    pub heights: HashMap<Txid, Option<Height>>,

    /// txids sorted by height descending, then txid descending (unconfirmed first)
    pub sorted_txids: Vec<Txid>,

    /// Wallet unspent outpoints and their script pubkeys
    pub unspent: HashMap<OutPoint, Script>,

    /// For every outpoint spent by a transaction that passed through this cache, the
    /// txids of the transactions spending it. Built from transaction bodies as they are
    /// added (see [`Self::extend_all_txs`]) and never pruned; whether a spend is LIVE is
    /// decided by the spender still being in `heights`. This is what makes the unspent
    /// set independent of the order and batching of updates: an output is unspent only
    /// while no live wallet transaction spends it, regardless of which update mentions
    /// which txid.
    spent_by: HashMap<OutPoint, HashSet<Txid>>,

    /// unblinded values
    pub unblinded: HashMap<OutPoint, TxOutSecrets>,

    /// txids of wallet txs where none of the wallet-owned inputs or outputs could be
    /// unblinded, i.e. `unblinded` has no entry for any of their relevant outpoints.
    ///
    /// Such transactions carry no balance information and are excluded by default from
    /// [`crate::Wollet::txs()`].
    cannot_unblind_txids: HashSet<Txid>,

    /// height and hash of tip of the blockchain
    pub tip: (Height, BlockHash),

    /// Contains the time of blocks at the given height. There are only heights containinig wallet txs
    pub timestamps: HashMap<Height, Timestamp>,

    /// last unused index for external addresses for current descriptor
    pub last_unused_external: AtomicU32,

    /// last unused index for internal addresses (changes) for current descriptor
    pub last_unused_internal: AtomicU32,
}

impl Default for Cache {
    fn default() -> Self {
        Self {
            txs_store: Arc::new(MemoryStore::default()),
            txids: HashSet::default(),
            paths: HashMap::default(),
            scripts: HashMap::default(),
            heights: HashMap::default(),
            sorted_txids: vec![],
            unspent: HashMap::new(),
            spent_by: HashMap::default(),
            unblinded: HashMap::default(),
            cannot_unblind_txids: HashSet::default(),
            tip: (0, BlockHash::all_zeros()),
            last_unused_internal: 0.into(),
            last_unused_external: 0.into(),
            timestamps: HashMap::default(),
        }
    }
}

impl std::hash::Hash for Cache {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let mut vec: Vec<_> = self.all_txids().iter().collect();
        vec.sort();
        vec.hash(state);

        let mut vec: Vec<_> = self.paths.iter().collect();
        vec.sort();
        vec.hash(state);

        // We don't hash the blinding public key for backward compatibility reasons
        let mut vec: Vec<_> = self.scripts.iter().map(|(k, v)| (k, &v.0)).collect();
        vec.sort();
        vec.hash(state);

        let mut vec: Vec<_> = self.heights.iter().collect();
        vec.sort();
        vec.hash(state);

        let mut vec: Vec<_> = self.unblinded.iter().collect();
        vec.sort_by_key(|kv| kv.0);
        vec.hash(state);

        self.tip.hash(state);

        let mut vec: Vec<_> = self.timestamps.iter().collect();
        vec.sort();
        vec.hash(state);

        self.last_unused_external
            .load(Ordering::Relaxed)
            .hash(state);

        self.last_unused_internal
            .load(Ordering::Relaxed)
            .hash(state);
    }
}

#[derive(Default, Debug)]
pub struct ScriptBatch {
    pub cached: bool,
    #[allow(clippy::type_complexity)]
    pub value: Vec<(Script, (Chain, ChildNumber, Option<BlindingPublicKey>))>,
}

impl Cache {
    pub fn new(txs_store: Arc<dyn DynStore>) -> Self {
        Self {
            txs_store,
            ..Self::default()
        }
    }

    pub fn get_script_batch(
        &self,
        batch: u32,
        descriptor: &WolletDescriptor,
        ext_int: Chain,
    ) -> Result<ScriptBatch, Error> {
        let mut result = ScriptBatch {
            cached: true,
            ..Default::default()
        };

        // For Spks, return all scripts in batch 0, empty for subsequent batches
        if let Some(count) = descriptor.spk_count() {
            if batch > 0 {
                return Ok(result);
            }
            for j in 0..count as u32 {
                let child = ChildNumber::from_normal_idx(j)?;
                let (script, blinding_pubkey, cached) =
                    self.get_or_derive(Chain::External, child, descriptor)?;
                result.cached = cached;
                result
                    .value
                    .push((script, (Chain::External, child, blinding_pubkey)));
            }
            return Ok(result);
        }

        let start = batch * BATCH_SIZE;
        let end = start + BATCH_SIZE;
        for j in start..end {
            let child = ChildNumber::from_normal_idx(j)?;
            let (script, blinding_pubkey, cached) =
                self.get_or_derive(ext_int, child, descriptor)?;
            result.cached = cached;
            result
                .value
                .push((script, (ext_int, child, blinding_pubkey)));
        }

        Ok(result)
    }

    pub fn get_or_derive(
        &self,
        ext_int: Chain,
        child: ChildNumber,
        descriptor: &WolletDescriptor,
    ) -> Result<(Script, Option<BlindingPublicKey>, bool), Error> {
        let opt_script = self.scripts.get(&(ext_int, child));
        let (script, blinding_pubkey, cached) = match opt_script {
            Some((script, blinding_pubkey)) => (script.clone(), *blinding_pubkey, true),
            None => {
                let (script, blinding_pubkey) =
                    descriptor.derive_script_and_blinding_key(ext_int, child)?;
                (script, blinding_pubkey, false)
            }
        };
        Ok((script, blinding_pubkey, cached))
    }

    pub fn sorted_txids(&self) -> impl Iterator<Item = (&Txid, &Option<Height>)> {
        self.sorted_txids.iter().map(|txid| {
            let height = self.heights.get(txid).unwrap_or(&None);
            (txid, height)
        })
    }

    pub fn unspent(&self) -> &HashMap<OutPoint, Script> {
        &self.unspent
    }

    pub fn tx_height(&self, txid: &Txid) -> Option<&Option<Height>> {
        self.heights.get(txid)
    }

    pub fn heights(&self) -> &HashMap<Txid, Option<Height>> {
        &self.heights
    }

    pub fn all_txs(&self) -> impl Iterator<Item = (Txid, Transaction)> + '_ {
        self.all_txids()
            .iter()
            .filter_map(|&txid| self.tx(&txid).map(|tx| (txid, tx)))
    }

    pub fn tx(&self, txid: &Txid) -> Option<Transaction> {
        // TODO: return result to handle the case where the store errors
        let bytes = self.txs_store.get(&tx_key(txid)).ok()??;
        elements::encode::deserialize(&bytes).ok()
    }

    /// Return the transaction from the in-memory `txs` slice when available, and
    /// fall back to the cache/store otherwise.
    ///
    /// This avoids hitting `self.tx()` for transactions downloaded in the current
    /// update, which can be more expensive when it reads from disk. The fallback is
    /// still required because some callers pass txids for already-known
    /// transactions, for example when only their confirmation height changed.
    pub(crate) fn tx_as_fallback(
        &self,
        txid: &Txid,
        txs: &[(Txid, Transaction)],
    ) -> Option<Transaction> {
        // Usually txs is small thus we do a linear search instead of building an HashMap
        txs.iter()
            .find(|(candidate, _)| candidate == txid)
            .map(|(_, tx)| tx.clone())
            .or_else(|| self.tx(txid))
    }

    fn outpoint_script(&self, outpoint: &OutPoint, txs: &[(Txid, Transaction)]) -> Option<Script> {
        self.tx_as_fallback(&outpoint.txid, txs).and_then(|tx| {
            tx.output
                .get(outpoint.vout as usize)
                .map(|txout| txout.script_pubkey.clone())
        })
    }

    pub fn all_txids(&self) -> &HashSet<Txid> {
        &self.txids
    }

    pub fn add_txids_from_txs_store(&mut self) {
        let txids: HashSet<Txid> = self
            .txs_store
            .get(TXIDS_KEY)
            .ok()
            .flatten()
            .and_then(|b| serde_json::from_slice::<Vec<String>>(&b).ok())
            .unwrap_or_default()
            .into_iter()
            .filter_map(|s| s.parse().ok())
            .collect();
        self.txids.extend(txids);
    }

    /// Load the "cannot unblind" txids computed in previous sessions.
    ///
    /// This is a single store read of a persisted snapshot (see [`Self::persist_cannot_unblind`]),
    /// as opposed to recomputing it by fetching every transaction from `txs_store`.
    pub fn add_cannot_unblind_from_store(&mut self) {
        let txids: HashSet<Txid> = self
            .txs_store
            .get(CANNOT_UNBLIND_TXIDS_KEY)
            .ok()
            .flatten()
            .and_then(|b| serde_json::from_slice::<Vec<String>>(&b).ok())
            .unwrap_or_default()
            .into_iter()
            .filter_map(|s| s.parse().ok())
            .collect();
        self.cannot_unblind_txids.extend(txids);
    }

    /// Persist the current "cannot unblind" txids snapshot, so it can be loaded back with
    /// [`Self::add_cannot_unblind_from_store`] without recomputing it from `txs_store`.
    fn persist_cannot_unblind(&self) -> Result<(), Error> {
        let txids = serde_json::to_vec(
            &self
                .cannot_unblind_txids
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>(),
        )
        .map_err(Error::from)?;
        self.txs_store
            .put(CANNOT_UNBLIND_TXIDS_KEY, &txids)
            .map_err(Error::StoreError)
    }

    pub fn txs_store_is_persisted(&self) -> bool {
        self.txs_store.is_persisted()
    }

    fn rebuild_sorted_txids(&mut self) {
        let mut sorted: Vec<Txid> = self.heights.keys().cloned().collect();
        sorted.sort_by(|a, b| {
            // cannot panic here, sorted is heights keys
            let ha = self.heights[a].unwrap_or(u32::MAX);
            let hb = self.heights[b].unwrap_or(u32::MAX);
            hb.cmp(&ha).then(b.cmp(a))
        });
        self.sorted_txids = sorted;
    }

    /// Whether `outpoint` is spent by a wallet transaction that is currently live
    /// (present in `heights`, i.e. seen and not deleted).
    fn spent_by_live_tx(&self, outpoint: &OutPoint) -> bool {
        self.spent_by
            .get(outpoint)
            .is_some_and(|spenders| spenders.iter().any(|t| self.heights.contains_key(t)))
    }

    /// Maintain the unspent set from a delta update.
    ///
    /// Updates list a txid whenever its HEIGHT changes, not only when it is first seen,
    /// and a scan of a busy wallet is not an atomic snapshot: a block can land between
    /// the history fetches of two scripts, so a child can be reported confirmed while
    /// its parent is still reported unconfirmed. On the next scan the parent is listed
    /// alone (its height changed) and the child is not (its height is final). Re-adding
    /// the parent's outputs and removing only the inputs of the transactions in THIS
    /// update then resurrects the output the child spent — permanently, because the
    /// child is never listed again. Found by `fuzz_incremental_unspent_matches_derived`
    /// (seed 0, four steps in) after SideSwap customers saw balances too high by
    /// outputs spent months earlier.
    ///
    /// So a re-added output is checked against every live spender the cache knows
    /// (`spent_by`), not just the transactions carried by the update, and the whole set
    /// is swept against live spenders at the end so a phantom that reached persisted
    /// state heals on the next update or restore.
    fn update_unspent(
        &mut self,
        txid_height_new: &[(Txid, Option<u32>)],
        deleted_txids: &[Txid],
        new_txs: &[(Txid, Transaction)],
    ) {
        let txids_new: HashSet<&Txid> = txid_height_new.iter().map(|(txid, _)| txid).collect();

        let outputs_new: Vec<(OutPoint, Script)> = self
            .unblinded
            .keys()
            .filter(|op| txids_new.contains(&op.txid))
            .filter(|op| !self.spent_by_live_tx(op))
            .filter_map(|op| {
                self.outpoint_script(op, new_txs)
                    .map(|script| (*op, script))
            })
            .filter(|(_, script)| self.paths.contains_key(script))
            .collect();

        let inputs_new: HashSet<OutPoint> = txids_new
            .iter()
            .filter_map(|txid| self.tx_as_fallback(txid, new_txs))
            .flat_map(|tx| tx.input.into_iter().map(|i| i.previous_output))
            .collect();

        let inputs_to_restore: Vec<(OutPoint, Script)> = deleted_txids
            .iter()
            // Merged updates can still carry the full transaction in `new_txs` even
            // when the resulting update marks its txid as deleted.
            .filter_map(|txid| self.tx_as_fallback(txid, new_txs))
            .flat_map(|tx| tx.input.into_iter().map(|i| i.previous_output))
            .filter(|op| self.unblinded.contains_key(op))
            // another live transaction may spend the same output (a replacement)
            .filter(|op| !self.spent_by_live_tx(op))
            .filter_map(|op| {
                self.outpoint_script(&op, new_txs)
                    .map(|script| (op, script))
            })
            .filter(|(_, script)| self.paths.contains_key(script))
            .collect();

        // Add outputs of new txs
        self.unspent.extend(outputs_new);
        // Add inputs of deleted txs (they are utxos now)
        self.unspent.extend(inputs_to_restore);
        // Remove inputs of new txs (they are spent now)
        self.unspent.retain(|o, _| !inputs_new.contains(o));
        // Remove outputs of deleted txs (after adding inputs, so that an output spent
        // by another deleted tx does not remain in unspent)
        self.unspent
            .retain(|o, _| deleted_txids.iter().all(|txid| txid != &o.txid));
        // Sweep: nothing spent by a live wallet transaction is unspent, whatever the
        // history of updates that led here. Heals persisted phantoms on restore.
        let phantoms: Vec<OutPoint> = self
            .unspent
            .keys()
            .filter(|op| self.spent_by_live_tx(op))
            .copied()
            .collect();
        for op in phantoms {
            self.unspent.remove(&op);
        }
    }

    fn update_unspent_from_snapshot(
        &mut self,
        unspent: Vec<(OutPoint, Script)>,
        txs: &[(Txid, Transaction)],
    ) {
        let unspent = unspent
            .into_iter()
            .filter_map(|(op, script)| {
                self.unspent
                    .get(&op)
                    .cloned()
                    .or_else(|| {
                        if script.is_empty() {
                            None
                        } else {
                            Some(script)
                        }
                    })
                    .or_else(|| self.outpoint_script(&op, txs))
                    .map(|script| (op, script))
            })
            .collect();
        self.unspent = unspent;
    }

    fn update_heights(&mut self, new: &[(Txid, Option<u32>)], to_delete: &[Txid]) {
        self.heights.retain(|k, _| !to_delete.contains(k));
        self.heights.extend(new.iter().copied());
    }

    /// Set whether `tx` cannot be unblinded, i.e. none of its outputs nor the previous
    /// outputs of its inputs have an entry in `self.unblinded`.
    ///
    /// Returns whether this actually changed `self.cannot_unblind_txids`.
    fn set_cannot_unblind(&mut self, txid: &Txid, tx: &Transaction) -> bool {
        let has_unblinded = (0..tx.output.len()).any(|vout| {
            self.unblinded
                .contains_key(&OutPoint::new(*txid, vout as u32))
        }) || tx
            .input
            .iter()
            .any(|i| self.unblinded.contains_key(&i.previous_output));
        if has_unblinded {
            self.cannot_unblind_txids.remove(txid)
        } else {
            self.cannot_unblind_txids.insert(*txid)
        }
    }

    /// Update the "cannot unblind" txids using only the given `txs`, without touching
    /// `txs_store` for reads, and persist the resulting snapshot if it changed.
    ///
    /// This is called on every applied update, including when replaying persisted updates
    /// on wallet restore. In that case `txs` is empty, since the transactions are not
    /// carried by the update once they've already been persisted (see
    /// [`crate::update::UpdatesPersister::persist`]) and this is a no-op: restore instead
    /// loads the up to date snapshot in one shot via [`Self::add_cannot_unblind_from_store`],
    /// which avoids a `txs_store` read per transaction.
    fn update_cannot_unblind(&mut self, txs: &[(Txid, Transaction)]) -> Result<(), Error> {
        let mut changed = false;
        for (txid, tx) in txs {
            changed |= self.set_cannot_unblind(txid, tx);
        }
        if changed {
            self.persist_cannot_unblind()?;
        }
        Ok(())
    }

    /// Recompute the "cannot unblind" status of the given `txids`, fetching each transaction
    /// from the store, and persist the resulting snapshot if it changed.
    ///
    /// Used after unblinding info changes outside of the normal update flow, e.g.
    /// [`crate::Wollet::reunblind()`]. Unlike [`Self::update_cannot_unblind`] this does hit
    /// `txs_store`, which is fine since callers of this already scan every wallet transaction.
    pub(crate) fn refresh_cannot_unblind(
        &mut self,
        txids: impl IntoIterator<Item = Txid>,
    ) -> Result<(), Error> {
        let mut changed = false;
        for txid in txids {
            if let Some(tx) = self.tx(&txid) {
                changed |= self.set_cannot_unblind(&txid, &tx);
            }
        }
        if changed {
            self.persist_cannot_unblind()?;
        }
        Ok(())
    }

    /// Whether `txid` cannot be unblinded, i.e. none of its wallet-owned inputs or outputs
    /// could be unblinded.
    pub fn cannot_unblind(&self, txid: &Txid) -> bool {
        self.cannot_unblind_txids.contains(txid)
    }

    fn extend_all_txs(&mut self, txs: &[(Txid, Transaction)]) -> Result<(), Error> {
        for (txid, tx) in txs {
            self.txs_store
                .put(&tx_key(txid), &elements::encode::serialize(tx))
                .map_err(Error::StoreError)?;
            self.txids.insert(*txid);
            for input in &tx.input {
                self.spent_by
                    .entry(input.previous_output)
                    .or_default()
                    .insert(*txid);
            }
        }
        if !txs.is_empty() {
            // TODO: order keys
            let txids =
                serde_json::to_vec(&self.txids.iter().map(|t| t.to_string()).collect::<Vec<_>>())
                    .map_err(Error::from)?;
            self.txs_store
                .put(TXIDS_KEY, &txids)
                .map_err(Error::StoreError)?;
        }
        Ok(())
    }

    pub fn update(
        &mut self,
        txid_height_new: &[(Txid, Option<u32>)],
        deleted_txids: &[Txid],
        txs: &[(Txid, Transaction)],
        utxo_only: bool,
        unspent: Vec<(OutPoint, Script)>,
        use_unspent_snapshot: bool,
    ) -> Result<(), Error> {
        // TODO: cleanup this functions
        self.extend_all_txs(txs)?;
        self.update_heights(txid_height_new, deleted_txids);
        // Unlike client delta updates, a persisted v5 update is a snapshot and
        // must reconstruct txids without transaction payloads, which are already
        // available in the txs store.
        if use_unspent_snapshot {
            self.txids.extend(deleted_txids.iter().copied());
            self.txids
                .extend(txid_height_new.iter().map(|(txid, _)| *txid));
        }
        self.update_cannot_unblind(txs)?;
        self.rebuild_sorted_txids();
        if use_unspent_snapshot || utxo_only {
            self.update_unspent_from_snapshot(unspent, txs);
        } else {
            self.update_unspent(txid_height_new, deleted_txids, txs);
        }
        Ok(())
    }

    pub fn get_unblinded(&self, outpoint: &OutPoint) -> Option<&TxOutSecrets> {
        self.unblinded.get(outpoint)
    }

    pub fn all_unblinded(&self) -> &HashMap<OutPoint, TxOutSecrets> {
        &self.unblinded
    }

    pub fn extend_unblinded(
        &mut self,
        unblinded: impl IntoIterator<Item = (OutPoint, TxOutSecrets)>,
    ) {
        self.unblinded.extend(unblinded);
    }
}

#[cfg(test)]
mod tests {
    use crate::{cache::Cache, WolletDescriptor};
    use elements::{Address, AddressParams, Transaction, Txid};
    use elements_miniscript::ConfidentialDescriptor;
    use std::{
        collections::hash_map::DefaultHasher,
        convert::TryInto,
        hash::{Hash, Hasher},
        str::FromStr,
    };
    use tempfile::TempDir;

    #[test]
    fn test_address_derivation() {
        let tempdir = TempDir::new().unwrap();
        let mut dir = tempdir.path().to_path_buf();
        dir.push("store");
        let xpub = "tpubDD7tXK8KeQ3YY83yWq755fHY2JW8Ha8Q765tknUM5rSvjPcGWfUppDFMpQ1ScziKfW3ZNtZvAD7M3u7bSs7HofjTD3KP3YxPK7X6hwV8Rk2";
        let master_blinding_key =
            "9c8e4f05c7711a98c838be228bcb84924d4570ca53f35fa1c793e58841d47023";
        let checksum = "8w7cjcha";
        let desc_str = format!("ct(slip77({master_blinding_key}),elwpkh({xpub}/*))#{checksum}");
        let desc = ConfidentialDescriptor::<_>::from_str(&desc_str).unwrap();
        let desc: WolletDescriptor = desc.try_into().unwrap();
        let addr1 = desc.address(0, &AddressParams::LIQUID_TESTNET).unwrap();

        let cache = Cache::default();

        let x = cache
            .get_script_batch(
                0,
                &desc.as_single_descriptors().unwrap()[0],
                crate::descriptor::Chain::External,
            )
            .unwrap();
        assert_eq!(format!("{:?}", x.value[0]), "(Script(OP_0 OP_PUSHBYTES_20 d11ef9e68385138627b09d52d6fe12662d049224), (External, Normal { index: 0 }, Some(PublicKey(0525054b498a69342d90750ed5e8f91cb6fb4da48735fd7011fdbcfc0e8edee1f0a30ed1e5c1d730e281b73f70f02dec2cbe20d0ac864d3d3d6942a02d66c6e3))))");
        assert_ne!(x.value[0], x.value[1]);
        let addr2 = Address::from_script(
            &x.value[0].0,
            x.value[0].1 .2,
            &AddressParams::LIQUID_TESTNET,
        )
        .unwrap();
        assert_eq!(addr1, addr2)
    }

    #[test]
    fn test_cache_hash() {
        let mut cache = Cache::default();
        let mut hasher = DefaultHasher::new();
        cache.hash(&mut hasher);
        assert_eq!(11565483422739161174, hasher.finish());

        cache
            .heights
            .insert(<Txid as elements::hashes::Hash>::all_zeros(), None);
        let mut hasher = DefaultHasher::new();
        cache.hash(&mut hasher);
        assert_eq!(12004253425667158821, hasher.finish());

        // TODO test other fields change the hash
    }

    #[test]
    fn test_v5_restore_keeps_deleted_txids() {
        let tx = Transaction {
            version: 2,
            lock_time: elements::LockTime::ZERO,
            input: vec![],
            output: vec![],
        };
        let txid = tx.txid();
        let mut live = Cache::default();
        live.update(&[(txid, None)], &[], &[(txid, tx)], false, vec![], false)
            .unwrap();
        live.update(&[], &[txid], &[], false, vec![], false)
            .unwrap();

        assert!(live.all_txids().contains(&txid));

        // A merged persisted update may only mention a historical transaction in
        // `txid_height_delete`. The transaction is still present in the txs store,
        // so restoring the authoritative v5 snapshot must reconstruct its txid.
        let mut restored = Cache::default();
        restored
            .update(&[], &[txid], &[], false, vec![], true)
            .unwrap();

        assert_eq!(live.all_txids(), restored.all_txids());
    }

    /// The incremental `unspent` set must equal the set DERIVED from the
    /// wallet's transactions (owned outputs minus inputs of known txs), for
    /// any scan-shaped sequence of updates: txs first seen unconfirmed or
    /// confirmed, confirmed later, chained on unconfirmed parents, dropped
    /// from the mempool and re-added. Also: replaying the recorded updates
    /// into a fresh cache (a restore) must reach the same set.
    #[test]
    fn fuzz_incremental_unspent_matches_derived() {
        use crate::descriptor::Chain;
        use crate::elements::TxOutSecrets;
        use crate::hashes::Hash as _;
        use elements::bitcoin::bip32::ChildNumber;
        use elements::confidential::{
            Asset, AssetBlindingFactor, Nonce, Value, ValueBlindingFactor,
        };
        use elements::{AssetId, OutPoint, Sequence, TxIn, TxOut};
        use rand::{rngs::StdRng, Rng, SeedableRng};
        use std::collections::HashSet;

        struct SimTx {
            tx: Transaction,
            height: Option<u32>,
            confirm_at: usize,
            dropped: bool,
        }

        let asset = AssetId::from_slice(&[7u8; 32]).unwrap();
        let scripts: Vec<elements::Script> = (0..6)
            .map(|i| {
                let mut b = vec![0x00, 0x14];
                b.extend(std::iter::repeat(i as u8 + 1).take(20));
                elements::Script::from(b)
            })
            .collect();
        let external = {
            let mut b = vec![0x00, 0x14];
            b.extend(std::iter::repeat(0xee).take(20));
            elements::Script::from(b)
        };
        let secrets = |v: u64| {
            TxOutSecrets::new(
                asset,
                AssetBlindingFactor::zero(),
                v,
                ValueBlindingFactor::zero(),
            )
        };

        let seeds: u64 = std::env::var("LWK_FUZZ_SEEDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(200);
        for seed in 0..seeds {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut cache = Cache::default();
            for (i, s) in scripts.iter().enumerate() {
                cache
                    .paths
                    .insert(s.clone(), (Chain::External, ChildNumber::from(i as u32)));
            }
            let mut sim: Vec<SimTx> = vec![];
            let mut sim_unspent: Vec<OutPoint> = vec![];
            let mut height = 100u32;
            let mut ext_counter = 0u32;
            type Rec = (
                Vec<(Txid, Option<u32>)>,
                Vec<Txid>,
                Vec<(Txid, Transaction)>,
                Vec<(OutPoint, TxOutSecrets)>,
            );
            let mut recorded: Vec<Rec> = vec![];

            for step in 0..60usize {
                // create 0..2 wallet txs spending our own unspent coins (or an external coin)
                let n_new = rng.gen_range(0..3);
                for _ in 0..n_new {
                    let mut inputs = vec![];
                    for _ in 0..rng.gen_range(0..3) {
                        if sim_unspent.is_empty() {
                            break;
                        }
                        let idx = rng.gen_range(0..sim_unspent.len());
                        inputs.push(sim_unspent.swap_remove(idx));
                    }
                    if inputs.is_empty() {
                        ext_counter += 1;
                        let mut b = [0u8; 32];
                        b[..4].copy_from_slice(&ext_counter.to_le_bytes());
                        b[31] = 0xaa;
                        inputs.push(OutPoint::new(Txid::from_slice(&b).unwrap(), 0));
                    }
                    let mut outputs = vec![];
                    for _ in 0..rng.gen_range(1..4) {
                        let spk = if rng.gen_bool(0.8) {
                            scripts[rng.gen_range(0..scripts.len())].clone()
                        } else {
                            external.clone()
                        };
                        outputs.push(TxOut {
                            asset: Asset::Explicit(asset),
                            value: Value::Explicit(rng.gen_range(1..1000)),
                            nonce: Nonce::Null,
                            script_pubkey: spk,
                            witness: Default::default(),
                        });
                    }
                    let tx = Transaction {
                        version: 2,
                        lock_time: elements::LockTime::ZERO,
                        input: inputs
                            .iter()
                            .map(|op| TxIn {
                                previous_output: *op,
                                is_pegin: false,
                                script_sig: elements::Script::new(),
                                sequence: Sequence::MAX,
                                asset_issuance: Default::default(),
                                witness: Default::default(),
                            })
                            .collect(),
                        output: outputs,
                    };
                    let txid = tx.txid();
                    for (v, o) in tx.output.iter().enumerate() {
                        if cache.paths.contains_key(&o.script_pubkey) {
                            sim_unspent.push(OutPoint::new(txid, v as u32));
                        }
                    }
                    let confirm_delay = rng.gen_range(0..4);
                    sim.push(SimTx {
                        tx,
                        height: None,
                        confirm_at: step + confirm_delay,
                        dropped: false,
                    });
                }
                // sometimes a childless unconfirmed tx falls out of the mempool
                if rng.gen_bool(0.15) {
                    let live_inputs: HashSet<OutPoint> = sim
                        .iter()
                        .filter(|t| !t.dropped)
                        .flat_map(|t| t.tx.input.iter().map(|i| i.previous_output))
                        .collect();
                    let candidates: Vec<usize> = sim
                        .iter()
                        .enumerate()
                        .filter(|(_, t)| !t.dropped && t.height.is_none())
                        .filter(|(_, t)| {
                            let txid = t.tx.txid();
                            !(0..t.tx.output.len())
                                .any(|v| live_inputs.contains(&OutPoint::new(txid, v as u32)))
                        })
                        .map(|(i, _)| i)
                        .collect();
                    if !candidates.is_empty() {
                        let i = candidates[rng.gen_range(0..candidates.len())];
                        sim[i].dropped = true;
                        let txid = sim[i].tx.txid();
                        sim_unspent.retain(|op| op.txid != txid);
                        // its inputs become spendable again (if they are ours)
                        let live_txids: HashSet<Txid> = sim
                            .iter()
                            .filter(|t| !t.dropped)
                            .map(|t| t.tx.txid())
                            .collect();
                        for op in sim[i].tx.input.iter().map(|i| i.previous_output) {
                            if live_txids.contains(&op.txid) {
                                sim_unspent.push(op);
                            }
                        }
                    }
                }
                // confirmations
                let mut any = false;
                for t in sim.iter_mut() {
                    if !t.dropped && t.height.is_none() && t.confirm_at <= step {
                        t.height = Some(height);
                        any = true;
                    }
                }
                if any {
                    height += 1;
                }

                // the update exactly as full_scan would build it
                let live_txids: HashSet<Txid> = sim
                    .iter()
                    .filter(|t| !t.dropped)
                    .map(|t| t.tx.txid())
                    .collect();
                let mut txid_height_new = vec![];
                let mut txs = vec![];
                let mut unblinds = vec![];
                for t in sim.iter().filter(|t| !t.dropped) {
                    let txid = t.tx.txid();
                    match cache.heights.get(&txid) {
                        Some(h) if *h == t.height => {}
                        _ => txid_height_new.push((txid, t.height)),
                    }
                    if !cache.txids.contains(&txid) {
                        for (v, o) in t.tx.output.iter().enumerate() {
                            if cache.paths.contains_key(&o.script_pubkey) {
                                let val = match o.value {
                                    Value::Explicit(x) => x,
                                    _ => 0,
                                };
                                unblinds.push((OutPoint::new(txid, v as u32), secrets(val)));
                            }
                        }
                        txs.push((txid, t.tx.clone()));
                    }
                }
                let txid_height_delete: Vec<Txid> = cache
                    .heights
                    .keys()
                    .filter(|k| !live_txids.contains(*k))
                    .cloned()
                    .collect();
                if txid_height_new.is_empty() && txid_height_delete.is_empty() && txs.is_empty() {
                    continue;
                }
                if std::env::var("LWK_FUZZ_TRACE").ok().as_deref() == Some(&seed.to_string()) {
                    let short = |t: &Txid| t.to_string()[..8].to_owned();
                    eprintln!("-- step {step}");
                    for (t, h) in &txid_height_new {
                        eprintln!("   height {} -> {:?}", short(t), h);
                    }
                    for t in &txid_height_delete {
                        eprintln!("   delete {}", short(t));
                    }
                    for (t, tx) in &txs {
                        eprintln!(
                            "   body   {} inputs {:?} outputs {}",
                            short(t),
                            tx.input
                                .iter()
                                .map(|i| format!("{}:{}", short(&i.previous_output.txid), i.previous_output.vout))
                                .collect::<Vec<_>>(),
                            tx.output.len()
                        );
                    }
                }
                cache.extend_unblinded(unblinds.iter().cloned());
                cache
                    .update(&txid_height_new, &txid_height_delete, &txs, false, vec![], false)
                    .unwrap();
                if std::env::var("LWK_FUZZ_TRACE").ok().as_deref() == Some(&seed.to_string()) {
                    let short = |t: &Txid| t.to_string()[..8].to_owned();
                    eprintln!(
                        "   unspent now {:?}",
                        cache.unspent.keys().map(|o| format!("{}:{}", short(&o.txid), o.vout)).collect::<Vec<_>>()
                    );
                }
                recorded.push((txid_height_new, txid_height_delete, txs, unblinds));

                // derived truth: owned outputs of live txs minus inputs of live txs
                let live: Vec<&SimTx> = sim.iter().filter(|t| !t.dropped).collect();
                let spent: HashSet<OutPoint> = live
                    .iter()
                    .flat_map(|t| t.tx.input.iter().map(|i| i.previous_output))
                    .collect();
                let mut truth = HashSet::new();
                for t in &live {
                    let txid = t.tx.txid();
                    for (v, o) in t.tx.output.iter().enumerate() {
                        let op = OutPoint::new(txid, v as u32);
                        if cache.paths.contains_key(&o.script_pubkey) && !spent.contains(&op) {
                            truth.insert(op);
                        }
                    }
                }
                let got: HashSet<OutPoint> = cache.unspent.keys().cloned().collect();
                assert_eq!(
                    got,
                    truth,
                    "seed {seed} step {step}: phantom {:?} missing {:?}",
                    got.difference(&truth).collect::<Vec<_>>(),
                    truth.difference(&got).collect::<Vec<_>>()
                );
            }

            // a restore replays the same updates into a fresh cache
            let mut restored = Cache::default();
            for (i, s) in scripts.iter().enumerate() {
                restored
                    .paths
                    .insert(s.clone(), (Chain::External, ChildNumber::from(i as u32)));
            }
            for (n, d, txs, ub) in &recorded {
                restored.extend_unblinded(ub.iter().cloned());
                restored.update(n, d, txs, false, vec![], false).unwrap();
            }
            let a: HashSet<OutPoint> = cache.unspent.keys().cloned().collect();
            let b: HashSet<OutPoint> = restored.unspent.keys().cloned().collect();
            assert_eq!(a, b, "seed {seed}: restore differs from live");
        }
    }
}
