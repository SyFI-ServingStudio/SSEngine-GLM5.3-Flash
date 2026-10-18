//! The tier socket's messages (`runtime/transport.py`, msgspec msgpack): in, `TierBatch` (tagged map, its ops
//! array-like `TierMove`s) or `Shutdown`; out, `TierDone` and `WorkerError` (tagged maps, every field).

use rmpv::Value;

#[derive(Clone, Debug)]
pub struct Op {
    pub op_id: i64,
    pub op: String,
    pub tier: String,
    pub host_tier: Option<String>,
    pub b_lo: i64,
    pub b_hi: i64,
    pub block_start: i64,
    pub kv_blocks: Vec<Vec<i64>>,
    pub state_blocks: Vec<Vec<i64>>,
    pub slots: Vec<i64>,
    pub state_slots: Vec<i64>,
    pub host_slots: Vec<i64>,
    pub host_state: Vec<i64>,
    pub urgent: bool,
    pub recv_ns: Option<i64>, // when the receive thread decoded its batch (not on the wire)
}

pub enum Msg {
    Batch(Vec<Op>),
    Shutdown,
}

fn int(v: &Value) -> Result<i64, String> {
    v.as_i64().ok_or_else(|| format!("expected an int, got {v}"))
}

fn ints(v: &Value) -> Result<Vec<i64>, String> {
    v.as_array().ok_or_else(|| format!("expected an array, got {v}"))?.iter().map(int).collect()
}

fn lists(v: &Value) -> Result<Vec<Vec<i64>>, String> {
    v.as_array().ok_or_else(|| format!("expected an array, got {v}"))?.iter().map(ints).collect()
}

fn string(v: &Value) -> Result<String, String> {
    v.as_str().map(str::to_string).ok_or_else(|| format!("expected a str, got {v}"))
}

fn op(v: &Value) -> Result<Op, String> {
    let a = v.as_array().ok_or("TierMove: expected an array")?;
    if a.len() < 13 {
        return Err(format!("TierMove: {} fields", a.len()));
    }
    Ok(Op {
        op_id: int(&a[0])?,
        op: string(&a[1])?,
        tier: string(&a[2])?,
        host_tier: if a[3].is_nil() { None } else { Some(string(&a[3])?) },
        b_lo: int(&a[4])?,
        b_hi: int(&a[5])?,
        block_start: int(&a[6])?,
        kv_blocks: lists(&a[7])?,
        state_blocks: lists(&a[8])?,
        slots: ints(&a[9])?,
        state_slots: ints(&a[10])?,
        host_slots: ints(&a[11])?,
        host_state: ints(&a[12])?,
        urgent: a.get(13).map_or(Ok(false), |u| u.as_bool().ok_or("TierMove.urgent: expected a bool"))?,
        recv_ns: None,
    })
}

pub fn decode(buf: &[u8]) -> Result<Msg, String> {
    let v = rmpv::decode::read_value(&mut &buf[..]).map_err(|e| format!("tier message: {e}"))?;
    let m = v.as_map().ok_or("tier message: expected a tagged map")?;
    let get = |k: &str| m.iter().find(|(a, _)| a.as_str() == Some(k)).map(|(_, b)| b);
    match get("type").and_then(|t| t.as_str()) {
        Some("Shutdown") => Ok(Msg::Shutdown),
        Some("TierBatch") => {
            let ops = get("ops").and_then(|o| o.as_array()).ok_or("TierBatch without ops")?;
            Ok(Msg::Batch(ops.iter().map(op).collect::<Result<_, _>>()?))
        }
        t => Err(format!("tier message of type {t:?}")),
    }
}

#[derive(Clone, Debug)]
pub struct Done {
    pub stage: i64,
    pub op_id: i64,
    pub start_ns: i64,
    pub end_ns: i64,
    pub nbytes: i64,
    pub submit_ns: i64,
    pub gpu_ns: Option<i64>,
    pub mark_ns: Option<i64>,
    pub recv_ns: Option<i64>,
}

fn opt(v: Option<i64>) -> Value {
    v.map_or(Value::Nil, Value::from)
}

fn tagged(name: &str, fields: Vec<(&str, Value)>) -> Vec<u8> {
    let mut m = vec![(Value::from("type"), Value::from(name))];
    m.extend(fields.into_iter().map(|(k, v)| (Value::from(k), v)));
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &Value::Map(m)).expect("encode to a Vec");
    out
}

pub fn encode_done(d: &Done) -> Vec<u8> {
    tagged("TierDone", vec![
        ("stage", d.stage.into()),
        ("op_id", d.op_id.into()),
        ("start_ns", d.start_ns.into()),
        ("end_ns", d.end_ns.into()),
        ("nbytes", d.nbytes.into()),
        ("submit_ns", d.submit_ns.into()),
        ("gpu_ns", opt(d.gpu_ns)),
        ("mark_ns", opt(d.mark_ns)),
        ("recv_ns", opt(d.recv_ns)),
    ])
}

pub fn encode_error(stage: i64, error: &str) -> Vec<u8> {
    tagged("WorkerError", vec![("stage", stage.into()), ("error", error.into())])
}
