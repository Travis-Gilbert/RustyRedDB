use std::collections::BTreeMap;
use std::sync::Arc;

use crate::graph::error::GraphStoreResult;
use crate::tensor_derivation::TensorBlockPayloadStore;

#[derive(Clone)]
pub(crate) struct VectorPayloadStoreHandle {
    pub(crate) inner: Arc<dyn TensorBlockPayloadStore>,
}

impl std::fmt::Debug for VectorPayloadStoreHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VectorPayloadStoreHandle")
            .finish_non_exhaustive()
    }
}

impl VectorPayloadStoreHandle {
    pub(crate) fn new(inner: Arc<dyn TensorBlockPayloadStore>) -> Self {
        Self { inner }
    }

    pub(crate) fn get(&self) -> Arc<dyn TensorBlockPayloadStore> {
        Arc::clone(&self.inner)
    }
}

#[derive(Debug, Default)]
pub(crate) struct PreparedVectorProperties {
    pub(crate) packed_by_property: BTreeMap<String, Vec<u8>>,
    pub(crate) acquired_pin_receipts: Vec<VectorPayloadPinReceipt>,
}

#[derive(Clone, Debug)]
pub(crate) struct VectorPayloadPinReceipt {
    pub(crate) content_id: String,
    pub(crate) publication_key: String,
    pub(crate) pending_released: bool,
    pub(crate) claimed: bool,
}

pub(crate) struct VectorPayloadPinGuard {
    store: Arc<dyn TensorBlockPayloadStore>,
    receipts: Vec<VectorPayloadPinReceipt>,
    armed: bool,
}

impl VectorPayloadPinGuard {
    pub(crate) fn new(store: Arc<dyn TensorBlockPayloadStore>) -> Self {
        Self {
            store,
            receipts: Vec::new(),
            armed: true,
        }
    }

    pub(crate) fn extend(&mut self, receipts: impl IntoIterator<Item = VectorPayloadPinReceipt>) {
        self.receipts.extend(receipts);
    }

    fn release_receipt(&mut self, index: usize) -> GraphStoreResult<()> {
        let receipt = &mut self.receipts[index];
        if !receipt.pending_released {
            self.store.release_bound_tensor_block_pending(
                &receipt.content_id,
                &receipt.publication_key,
            )?;
            receipt.pending_released = true;
        }
        if receipt.claimed {
            self.store
                .release_claimed_tensor_block_pin(&receipt.content_id)?;
            receipt.claimed = false;
        }
        Ok(())
    }

    fn release_all(&mut self) -> GraphStoreResult<()> {
        while !self.receipts.is_empty() {
            let index = self.receipts.len() - 1;
            self.release_receipt(index)?;
            self.receipts.pop();
        }
        Ok(())
    }

    pub(crate) fn cancel(mut self) -> GraphStoreResult<()> {
        self.release_all()?;
        self.armed = false;
        Ok(())
    }

    pub(crate) fn finish_after_publication(mut self) {
        // The graph/catalog root is already durable. Cleanup is therefore
        // non-transactional: retain a failed component so Drop retries it,
        // but never report an apparent transaction failure to the caller.
        if self.release_all().is_ok() {
            self.armed = false;
        }
    }

    pub(crate) fn into_receipts(mut self) -> Vec<VectorPayloadPinReceipt> {
        self.armed = false;
        std::mem::take(&mut self.receipts)
    }
}

impl Drop for VectorPayloadPinGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        while !self.receipts.is_empty() {
            let index = self.receipts.len() - 1;
            let _ = self.release_receipt(index);
            self.receipts.pop();
        }
    }
}
