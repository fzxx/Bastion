// Bastion 棱堡 算法库

(function (root, factory) {
    if (typeof module === 'object' && module.exports) {
        // Node 环境
        module.exports = factory();
    } else {
        // 浏览器环境，挂到全局
        root.Bastion = factory();
    }
}(typeof self !== 'undefined' ? self : this, function () {

// 常量

const BLOCK_SIZE = 32;        // 分组大小，256 位
const KEY_SIZE = 32;          // 密钥固定 32 字节
const NONCE_SIZE = 24;        // nonce，CTR/AEAD 都用这个，192 位
const TAG_SIZE = 16;          // AEAD 认证标签，128 位
const ROUNDS = 16;            // 默认轮数，别乱改
const ARX_ROUNDS = ROUNDS;    // ARX 压缩函数的轮数，与分组密码轮数一致

// 旋转常数是算法定死的，相当于这套算法的"配方"，改一个安全性就变了
const GM_R1 = 13, GM_R2 = 19, GM_R3 = 23, GM_R4 = 29;
const GM_R5 = 7,  GM_R6 = 11, GM_R7 = 5,  GM_R8 = 15;
const GM_R9 = 17;  // 折回 XOR 用的旋转量
const HM_R1 = 9,  HM_R2 = 17, HM_R3 = 21, HM_R4 = 27;
const HM_R5 = 3,  HM_R6 = 11, HM_R7 = 7,  HM_R8 = 13;
const HM_R9 = 25;

const GF8_POLY = 0x2D;  // GF(2^8) 不可约多项式 x^8 + x^5 + x^3 + x^2 + 1

// 4 个 S 盒的 delta，密钥扩展用 S0-S3
const SBOX_DELTAS = [0x00, 0x5A, 0xB4, 0x2D];

// SHA-256 风格的初始 IV，素数平方根小数部分
const ARX_INIT_STATE = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]);

// 工具函数

// 把字符串统一编码成 UTF-8 字节，跟 Go/C/Rust 的 []byte 语义对齐；
// 已经是 Uint8Array 的直接返回，避免重复转换
function toBytes(x) {
    if (typeof x === 'string') return new TextEncoder().encode(x);
    return x;
}

// 入参规范化 + 类型校验：字符串先按 UTF-8 编码，其余必须是 Uint8Array。
// 普通数组/对象也能取出 length 和下标，但按位运算取值时会被静默转成 0 或 NaN，
// 结果是密钥、明文被悄悄换掉还不报错，所以这里直接拒绝
function toBytesStrict(x, name) {
    const b = toBytes(x);
    if (!(b instanceof Uint8Array)) {
        throw new TypeError(`${name} 必须是 Uint8Array 或字符串`);
    }
    return b;
}

// 长度/迭代次数校验：必须是有界安全整数，NaN/Infinity 这类值会造成空密钥或死循环
function requireSafeInt(value, name, min, max) {
    if (!Number.isSafeInteger(value) || value < min || (max !== undefined && value > max)) {
        const range = max === undefined ? `>= ${min}` : `${min} ~ ${max}`;
        throw new RangeError(`${name} 必须是 ${range} 的整数，当前为 ${value}`);
    }
    return value;
}

// 密钥材料用完就抹掉，减少内存里残留的明文密钥（JS 没有安全清零 API，尽力而为）
function wipe(...arrs) {
    for (const a of arrs) {
        if (a && typeof a.fill === 'function') a.fill(0);
    }
}

// 32 位无符号加法，>>> 0 转回无符号
function addU32(a, b) { return (a + b) >>> 0; }

// 循环右移，加密方向统一用右旋
function rotr32(x, r) { return ((x >>> r) | (x << (32 - r))) >>> 0; }

// 循环左移，只在解密时用，是右旋的逆运算
function rotl32(x, r) { return ((x << r) | (x >>> (32 - r))) >>> 0; }

// xmx 乘法置换，MurmurHash3 的 finalizer，给常数增加非线性
function xmxFinalize(x) {
    x = (x ^ (x >>> 16)) >>> 0;
    // 乘完用 >>> 0 转回无符号
    x = Math.imul(x, 0x45d9f3b) >>> 0;
    x = (x ^ (x >>> 16)) >>> 0;
    x = Math.imul(x, 0x45d9f3b) >>> 0;
    x = (x ^ (x >>> 16)) >>> 0;
    return x >>> 0;
}

// 小端读 32 位
function readU32LE(buf, off) {
    return (buf[off] | (buf[off + 1] << 8) | (buf[off + 2] << 16) | (buf[off + 3] << 24)) >>> 0;
}

// 小端写 32 位
function writeU32LE(buf, off, v) {
    buf[off] = v & 0xff;
    buf[off + 1] = (v >>> 8) & 0xff;
    buf[off + 2] = (v >>> 16) & 0xff;
    buf[off + 3] = (v >>> 24) & 0xff;
}

// 小端写 64 位
function writeU64LE(buf, off, v) {
    const lo = Number(v & 0xffffffffn);
    const hi = Number((v >> 32n) & 0xffffffffn);
    writeU32LE(buf, off, lo);
    writeU32LE(buf, off + 4, hi);
}

// 常量时间比较，避免被计时攻击；长度差并进结果而不是提前返回，避免泄漏长度
function constantTimeCompare(a, b) {
    const n = Math.max(a.length, b.length);
    let r = a.length ^ b.length;
    for (let i = 0; i < n; i++) {
        const x = i < a.length ? a[i] : 0;
        const y = i < b.length ? b[i] : 0;
        r |= x ^ y;
    }
    return r === 0;
}

// S 盒构造

// GF(2^8) 乘法，多项式 0x12D
function gf8Mul(a, b) {
    let result = 0;
    for (let i = 0; i < 8; i++) {
        if (b & 1) result ^= a;
        const carry = a & 0x80;
        a = (a << 1) & 0xff;
        if (carry) a ^= GF8_POLY;
        b >>>= 1;
    }
    return result & 0xff;
}

// GF(2^8) 求逆，用 Fermat 小定理，0 的逆规定为 0
function gf8Inv(a) {
    if (a === 0) return 0;
    let result = 1;
    let base = a;
    // a^(254) = a^(-1)，因为 a^255 = 1
    for (let exp = 254; exp > 0; exp >>>= 1) {
        if (exp & 1) result = gf8Mul(result, base);
        base = gf8Mul(base, base);
    }
    return result;
}

// Bastion 仿射变换，非循环矩阵
function affineBastion(x) {
    const b0 = (x >>> 0) & 1, b1 = (x >>> 1) & 1, b2 = (x >>> 2) & 1, b3 = (x >>> 3) & 1;
    const b4 = (x >>> 4) & 1, b5 = (x >>> 5) & 1, b6 = (x >>> 6) & 1, b7 = (x >>> 7) & 1;

    const y0 = b7 ^ b5 ^ b4 ^ b3 ^ b1 ^ 1;
    const y1 = b7 ^ b6 ^ b4 ^ b3 ^ b2;
    const y2 = b6 ^ b5 ^ b3 ^ b2 ^ b1 ^ 1;
    const y3 = b5 ^ b4 ^ b2 ^ b1 ^ b0 ^ 1;
    const y4 = b7 ^ b4 ^ b3 ^ b1 ^ b0;
    const y5 = b7 ^ b6 ^ b3 ^ b2 ^ b0 ^ 1;
    const y6 = b6 ^ b5 ^ b2 ^ b1 ^ b0;
    const y7 = b7 ^ b5 ^ b4 ^ b0 ^ 1;

    return ((y0 & 1) | ((y1 & 1) << 1) | ((y2 & 1) << 2) | ((y3 & 1) << 3) |
            ((y4 & 1) << 4) | ((y5 & 1) << 5) | ((y6 & 1) << 6) | ((y7 & 1) << 7)) & 0xff;
}

// 生成基础 S 盒：S(x) = affine(inv(x))
function genSbox() {
    const out = new Uint8Array(256);
    for (let i = 0; i < 256; i++) out[i] = affineBastion(gf8Inv(i));
    return out;
}

// 预移位 S 盒查找表，省掉运行时移位
const sbox0T0 = new Uint32Array(256);
const sbox1T8 = new Uint32Array(256);
const sbox2T16 = new Uint32Array(256);
const sbox3T24 = new Uint32Array(256);

// 圆周率 π 派生的固定轮常数，取自 π 十六进制小数部分连续截取，每 8 个字符一个 32 位字
const roundConstants = [];
for (let r = 0; r < ROUNDS; r++) roundConstants.push(new Uint32Array(8));
(function fillRoundConstants() {
    const hex = "243F6A8885A308D313198A2E03707344A4093822299F31D0082EFA98EC4E6C89" +
                "452821E638D01377BE5466CF34E90C6CC0AC29B7C97C50DD3F84D5B5B5470917" +
                "9216D5D98979FB1BD1310BA698DFB5AC2FFD72DBD01ADFB7B8E1AFED6A267E96" +
                "BA7C9045F12C7F9924A19947B3916CF70801F2E2858EFC16636920D871574E69" +
                "A458FEA3F4933D7E0D95748F728EB658718BCD5882154AEE7B54A41DC25A59B5" +
                "9C30D5392AF26013C5D1B023286085F0CA417918B8DB38EF8E79DCB0603A180E" +
                "6C9E0E8BB01E8A3ED71577C1BD314B2778AF2FDA55605C60E65525F3AA55AB94" +
                "5748986263E8144055CA396A2AAB10B6B4CC5C341141E8CEA15486AF7C72E993" +
                "B3EE1411636FBC2A2BA9C55D741831F6CE5C3E169B87931EAFD6BA336C24CF5C" +
                "7A325381289586773B8F48986B4BB9AFC4BFE81B6628219361D809CCFB21A991" +
                "487CAC605DEC8032EF845D5DE98575B1DC262302EB651B8823893E81D396ACC5" +
                "0F6D6FF383F442392E0B4482A484200469C8F04A9E1F9B5E21C66842F6E96C9A" +
                "670C9C61ABD388F06A51A0D2D8542F68960FA728AB5133A36EEF0B6C137A3BE4" +
                "BA3BF0507EFB2A98A1F1651D39AF017666CA593E82430E888CEE8619456F9FB4" +
                "7D84A5C33B8B5EBEE06F75D885C12073401A449F56C16AA64ED3AA62363F7706" +
                "1BFEDF72429B023D37D0D724D00A1248DB0FEAD349F1C09B075372C980991B7B";
    for (let r = 0; r < ROUNDS; r++) {
        for (let j = 0; j < 8; j++) {
            roundConstants[r][j] = parseInt(hex.substring(r * 64 + j * 8, r * 64 + j * 8 + 8), 16) >>> 0;
        }
    }
})();

// 末块封口常数（π 派生），标记最后一块，阻断长度扩展
const finalConst = new Uint32Array([
    0x8F02C1BA, 0xB7ED2F35, 0x61711891, 0xDE7A9041,
    0x758EDD75, 0x47708A47, 0x07B0E7A5, 0x4B91FB25,
]);

// 非末块用的全零封口常数，全局共用一份
const zeroSeal = new Uint32Array(8);

// 对 uint32 用 S0-S3 做 S 盒替换，密钥扩展里用
function sboxWordApply32(v) {
    return (sbox3T24[(v >>> 24) & 0xff] |
            sbox2T16[(v >>> 16) & 0xff] |
            sbox1T8[(v >>> 8) & 0xff] |
            sbox0T0[v & 0xff]) >>> 0;
}

// 模块初始化，加载库时跑一次
function init() {
    const base = genSbox();
    // 4 个 S 盒各自异或一个 delta
    const sboxes = [new Uint8Array(256), new Uint8Array(256), new Uint8Array(256), new Uint8Array(256)];
    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 256; j++) {
            sboxes[i][j] = base[j] ^ SBOX_DELTAS[i];
        }
    }
    // 预移位表
    for (let i = 0; i < 256; i++) {
        sbox0T0[i]  = sboxes[0][i];
        sbox1T8[i]  = sboxes[1][i] << 8;
        sbox2T16[i] = sboxes[2][i] << 16;
        sbox3T24[i] = sboxes[3][i] << 24;
    }
}

init();

// 分组密码

// BlockCipher 实例，轮密钥存成 Uint32Array，方便按字访问
class BlockCipher {
    constructor(key) {
        // 字符串按 UTF-8 编码成字节，非字节类型直接拒绝，避免密钥被静默强转成别的值
        key = toBytesStrict(key, '密钥');
        if (key.length !== KEY_SIZE) {
            throw new Error(`密钥长度必须为 ${KEY_SIZE} 字节`);
        }
        this.rounds = ROUNDS;
        this.destroyed = false;  // 清零后置位，防止拿已销毁的实例继续加解密
        // 72 个 uint64 = 144 个 uint32
        this.roundKeys = new Uint32Array((ROUNDS + 2) * 8);
        this.expandKey(key);
    }

    // 密钥扩展，输出 144 个 uint32
    expandKey(key) {
        const w = this.roundKeys;
        const totalWords = (this.rounds + 2) * 8;
        for (let i = 0; i < 8; i++) w[i] = readU32LE(key, i * 4);

        for (let i = 8; i < totalWords; i++) {
            let temp = w[i - 1] >>> 0;
            if (i % 8 === 0) {
                temp = rotl32(temp, 8);
                temp = sboxWordApply32(temp);
            } else if (i % 4 === 0) {
                temp = sboxWordApply32(temp);
                temp = rotl32(temp, 16);
                temp = addU32(temp, rotl32(temp, 7));
            } else {
                temp = xmxFinalize(temp);
            }
            w[i] = (w[i - 8] ^ temp) >>> 0;
        }

        // 轮常数预合并进轮密钥，运行时少一次异或
        for (let r = 0; r < this.rounds; r++) {
            const base = (r + 1) * 8;  // 跳过输入白化 w[0..7]
            const sc = roundConstants[r];
            for (let j = 0; j < 8; j++) {
                w[base + j] = (w[base + j] ^ sc[j]) >>> 0;
            }
        }
    }

    // 单块加密，src 和 dst 都是 32 字节
    encryptBlock(src, dst) {
        // 入参不足 32 字节时读出来是 undefined（按位运算变 0）、写出会被静默丢弃，
        // 密文会错得无声无息，入口先卡长度
        if (!(src instanceof Uint8Array) || !(dst instanceof Uint8Array) ||
            src.length < BLOCK_SIZE || dst.length < BLOCK_SIZE) {
            throw new Error(`加解密输入输出都必须是不小于 ${BLOCK_SIZE} 字节的 Uint8Array`);
        }
        if (this.destroyed) {
            throw new Error('BlockCipher 已销毁，不能继续使用');
        }
        encryptBlockInlineRef(this,
            readU32LE(src, 0), readU32LE(src, 4), readU32LE(src, 8), readU32LE(src, 12),
            readU32LE(src, 16), readU32LE(src, 20), readU32LE(src, 24), readU32LE(src, 28),
            dst);
    }

    // 单块解密，加密的逆运算
    decryptBlock(src, dst) {
        if (!(src instanceof Uint8Array) || !(dst instanceof Uint8Array) ||
            src.length < BLOCK_SIZE || dst.length < BLOCK_SIZE) {
            throw new Error(`加解密输入输出都必须是不小于 ${BLOCK_SIZE} 字节的 Uint8Array`);
        }
        if (this.destroyed) {
            throw new Error('BlockCipher 已销毁，不能继续使用');
        }
        decryptBlockInlineRef(this,
            readU32LE(src, 0), readU32LE(src, 4), readU32LE(src, 8), readU32LE(src, 12),
            readU32LE(src, 16), readU32LE(src, 20), readU32LE(src, 24), readU32LE(src, 28),
            dst);
    }

    // 用完清零，清零后实例作废
    zeroize() {
        this.roundKeys.fill(0);
        this.rounds = 0;
        this.destroyed = true;
    }
}

// 加密主循环，结果直接写进 dst，省掉每块的闭包和临时数组分配
function encryptBlockInlineRef(c, s0, s1, s2, s3, s4, s5, s6, s7, dst) {
    const k = c.roundKeys;
    const rounds = c.rounds;

    // 输入白化
    s0 = (s0 ^ k[0]) >>> 0; s1 = (s1 ^ k[1]) >>> 0;
    s2 = (s2 ^ k[2]) >>> 0; s3 = (s3 ^ k[3]) >>> 0;
    s4 = (s4 ^ k[4]) >>> 0; s5 = (s5 ^ k[5]) >>> 0;
    s6 = (s6 ^ k[6]) >>> 0; s7 = (s7 ^ k[7]) >>> 0;

    let ki = 8;  // 轮密钥按 uint32 索引，跳过输入白化的 8 个字
    for (let r = 0; r < rounds; r++) {
        // ColumnMix：偶数轮 G_Mix，奇数轮 H_Mix
        if ((r & 1) === 0) {
            // G_Mix
            s0 = addU32(s0, rotr32(s2, GM_R5)); s1 = addU32(s1, rotr32(s3, GM_R5));
            s6 = (s6 ^ s0) >>> 0; s7 = (s7 ^ s1) >>> 0;
            s6 = rotr32(s6, GM_R1); s7 = rotr32(s7, GM_R1);
            s4 = addU32(s4, rotr32(s6, GM_R6)); s5 = addU32(s5, rotr32(s7, GM_R6));
            s2 = (s2 ^ s4) >>> 0; s3 = (s3 ^ s5) >>> 0;
            s2 = rotr32(s2, GM_R2); s3 = rotr32(s3, GM_R2);
            // 折回 XOR
            s0 = (s0 ^ rotr32(s4, GM_R9)) >>> 0; s1 = (s1 ^ rotr32(s5, GM_R9)) >>> 0;
            s0 = addU32(s0, rotr32(s2, GM_R7)); s1 = addU32(s1, rotr32(s3, GM_R7));
            s6 = (s6 ^ s0) >>> 0; s7 = (s7 ^ s1) >>> 0;
            s6 = rotr32(s6, GM_R3); s7 = rotr32(s7, GM_R3);
            s4 = addU32(s4, rotr32(s6, GM_R8)); s5 = addU32(s5, rotr32(s7, GM_R8));
            s2 = (s2 ^ s4) >>> 0; s3 = (s3 ^ s5) >>> 0;
            s2 = rotr32(s2, GM_R4); s3 = rotr32(s3, GM_R4);
        } else {
            // H_Mix，旋转量不同
            s0 = addU32(s0, rotr32(s2, HM_R5)); s1 = addU32(s1, rotr32(s3, HM_R5));
            s6 = (s6 ^ s0) >>> 0; s7 = (s7 ^ s1) >>> 0;
            s6 = rotr32(s6, HM_R1); s7 = rotr32(s7, HM_R1);
            s4 = addU32(s4, rotr32(s6, HM_R6)); s5 = addU32(s5, rotr32(s7, HM_R6));
            s2 = (s2 ^ s4) >>> 0; s3 = (s3 ^ s5) >>> 0;
            s2 = rotr32(s2, HM_R2); s3 = rotr32(s3, HM_R2);
            s0 = (s0 ^ rotr32(s4, HM_R9)) >>> 0; s1 = (s1 ^ rotr32(s5, HM_R9)) >>> 0;
            s0 = addU32(s0, rotr32(s2, HM_R7)); s1 = addU32(s1, rotr32(s3, HM_R7));
            s6 = (s6 ^ s0) >>> 0; s7 = (s7 ^ s1) >>> 0;
            s6 = rotr32(s6, HM_R3); s7 = rotr32(s7, HM_R3);
            s4 = addU32(s4, rotr32(s6, HM_R8)); s5 = addU32(s5, rotr32(s7, HM_R8));
            s2 = (s2 ^ s4) >>> 0; s3 = (s3 ^ s5) >>> 0;
            s2 = rotr32(s2, HM_R4); s3 = rotr32(s3, HM_R4);
        }

        // ShiftRows，单个 8-循环置换，循环 (0 2 7 1 5 4 3 6)
        let t = s0; s0 = s2; s2 = s7; s7 = s1;
        s1 = s5; s5 = s4; s4 = s3; s3 = s6; s6 = t;

        // AddRoundKey
        s0 = (s0 ^ k[ki]) >>> 0; s1 = (s1 ^ k[ki + 1]) >>> 0;
        s2 = (s2 ^ k[ki + 2]) >>> 0; s3 = (s3 ^ k[ki + 3]) >>> 0;
        s4 = (s4 ^ k[ki + 4]) >>> 0; s5 = (s5 ^ k[ki + 5]) >>> 0;
        s6 = (s6 ^ k[ki + 6]) >>> 0; s7 = (s7 ^ k[ki + 7]) >>> 0;
        ki += 8;
    }

    // 输出白化，算完直接写回目标缓冲
    writeU32LE(dst, 0, (s0 ^ k[ki]) >>> 0); writeU32LE(dst, 4, (s1 ^ k[ki + 1]) >>> 0);
    writeU32LE(dst, 8, (s2 ^ k[ki + 2]) >>> 0); writeU32LE(dst, 12, (s3 ^ k[ki + 3]) >>> 0);
    writeU32LE(dst, 16, (s4 ^ k[ki + 4]) >>> 0); writeU32LE(dst, 20, (s5 ^ k[ki + 5]) >>> 0);
    writeU32LE(dst, 24, (s6 ^ k[ki + 6]) >>> 0); writeU32LE(dst, 28, (s7 ^ k[ki + 7]) >>> 0);
}

// 解密主循环，加密的逆运算
function decryptBlockInlineRef(c, s0, s1, s2, s3, s4, s5, s6, s7, dst) {
    const k = c.roundKeys;
    const rounds = c.rounds;

    // 先解除输出白化
    let ki = 8 + rounds * 8;
    s0 = (s0 ^ k[ki]) >>> 0; s1 = (s1 ^ k[ki + 1]) >>> 0;
    s2 = (s2 ^ k[ki + 2]) >>> 0; s3 = (s3 ^ k[ki + 3]) >>> 0;
    s4 = (s4 ^ k[ki + 4]) >>> 0; s5 = (s5 ^ k[ki + 5]) >>> 0;
    s6 = (s6 ^ k[ki + 6]) >>> 0; s7 = (s7 ^ k[ki + 7]) >>> 0;

    for (let r = rounds - 1; r >= 0; r--) {
        ki = 8 + r * 8;
        // 解除 AddRoundKey
        s0 = (s0 ^ k[ki]) >>> 0; s1 = (s1 ^ k[ki + 1]) >>> 0;
        s2 = (s2 ^ k[ki + 2]) >>> 0; s3 = (s3 ^ k[ki + 3]) >>> 0;
        s4 = (s4 ^ k[ki + 4]) >>> 0; s5 = (s5 ^ k[ki + 5]) >>> 0;
        s6 = (s6 ^ k[ki + 6]) >>> 0; s7 = (s7 ^ k[ki + 7]) >>> 0;

        // 逆 ShiftRows，8-循环置换的逆
        let t = s0; s0 = s6; s6 = s3; s3 = s4;
        s4 = s5; s5 = s1; s1 = s7; s7 = s2; s2 = t;

        // 逆 ColumnMix
        if ((r & 1) === 0) {
            // 逆 G_Mix
            s2 = (rotl32(s2, GM_R4) ^ s4) >>> 0; s3 = (rotl32(s3, GM_R4) ^ s5) >>> 0;
            s4 = (s4 - rotr32(s6, GM_R8)) >>> 0; s5 = (s5 - rotr32(s7, GM_R8)) >>> 0;
            s6 = (rotl32(s6, GM_R3) ^ s0) >>> 0; s7 = (rotl32(s7, GM_R3) ^ s1) >>> 0;
            s0 = (s0 - rotr32(s2, GM_R7)) >>> 0; s1 = (s1 - rotr32(s3, GM_R7)) >>> 0;
            // 折回 XOR 是自逆的，直接再异或一次
            s0 = (s0 ^ rotr32(s4, GM_R9)) >>> 0; s1 = (s1 ^ rotr32(s5, GM_R9)) >>> 0;
            s2 = (rotl32(s2, GM_R2) ^ s4) >>> 0; s3 = (rotl32(s3, GM_R2) ^ s5) >>> 0;
            s4 = (s4 - rotr32(s6, GM_R6)) >>> 0; s5 = (s5 - rotr32(s7, GM_R6)) >>> 0;
            s6 = (rotl32(s6, GM_R1) ^ s0) >>> 0; s7 = (rotl32(s7, GM_R1) ^ s1) >>> 0;
            s0 = (s0 - rotr32(s2, GM_R5)) >>> 0; s1 = (s1 - rotr32(s3, GM_R5)) >>> 0;
        } else {
            // 逆 H_Mix
            s2 = (rotl32(s2, HM_R4) ^ s4) >>> 0; s3 = (rotl32(s3, HM_R4) ^ s5) >>> 0;
            s4 = (s4 - rotr32(s6, HM_R8)) >>> 0; s5 = (s5 - rotr32(s7, HM_R8)) >>> 0;
            s6 = (rotl32(s6, HM_R3) ^ s0) >>> 0; s7 = (rotl32(s7, HM_R3) ^ s1) >>> 0;
            s0 = (s0 - rotr32(s2, HM_R7)) >>> 0; s1 = (s1 - rotr32(s3, HM_R7)) >>> 0;
            s0 = (s0 ^ rotr32(s4, HM_R9)) >>> 0; s1 = (s1 ^ rotr32(s5, HM_R9)) >>> 0;
            s2 = (rotl32(s2, HM_R2) ^ s4) >>> 0; s3 = (rotl32(s3, HM_R2) ^ s5) >>> 0;
            s4 = (s4 - rotr32(s6, HM_R6)) >>> 0; s5 = (s5 - rotr32(s7, HM_R6)) >>> 0;
            s6 = (rotl32(s6, HM_R1) ^ s0) >>> 0; s7 = (rotl32(s7, HM_R1) ^ s1) >>> 0;
            s0 = (s0 - rotr32(s2, HM_R5)) >>> 0; s1 = (s1 - rotr32(s3, HM_R5)) >>> 0;
        }
    }

    // 解除输入白化，算完直接写回目标缓冲
    writeU32LE(dst, 0, (s0 ^ k[0]) >>> 0); writeU32LE(dst, 4, (s1 ^ k[1]) >>> 0);
    writeU32LE(dst, 8, (s2 ^ k[2]) >>> 0); writeU32LE(dst, 12, (s3 ^ k[3]) >>> 0);
    writeU32LE(dst, 16, (s4 ^ k[4]) >>> 0); writeU32LE(dst, 20, (s5 ^ k[5]) >>> 0);
    writeU32LE(dst, 24, (s6 ^ k[6]) >>> 0); writeU32LE(dst, 28, (s7 ^ k[7]) >>> 0);
}

// ARX 压缩函数

// ARX 的一轮变换，跟分组密码主循环共用 G/H Mix，但没有 S 盒和密钥
function arxRound(s, round) {
    // 就地读写工作数组，省掉每轮新建数组分配
    let s0 = s[0], s1 = s[1], s2 = s[2], s3 = s[3];
    let s4 = s[4], s5 = s[5], s6 = s[6], s7 = s[7];
    if ((round & 1) === 0) {
        s0 = addU32(s0, rotr32(s2, GM_R5)); s1 = addU32(s1, rotr32(s3, GM_R5));
        s6 = (s6 ^ s0) >>> 0; s7 = (s7 ^ s1) >>> 0;
        s6 = rotr32(s6, GM_R1); s7 = rotr32(s7, GM_R1);
        s4 = addU32(s4, rotr32(s6, GM_R6)); s5 = addU32(s5, rotr32(s7, GM_R6));
        s2 = (s2 ^ s4) >>> 0; s3 = (s3 ^ s5) >>> 0;
        s2 = rotr32(s2, GM_R2); s3 = rotr32(s3, GM_R2);
        s0 = (s0 ^ rotr32(s4, GM_R9)) >>> 0; s1 = (s1 ^ rotr32(s5, GM_R9)) >>> 0;
        s0 = addU32(s0, rotr32(s2, GM_R7)); s1 = addU32(s1, rotr32(s3, GM_R7));
        s6 = (s6 ^ s0) >>> 0; s7 = (s7 ^ s1) >>> 0;
        s6 = rotr32(s6, GM_R3); s7 = rotr32(s7, GM_R3);
        s4 = addU32(s4, rotr32(s6, GM_R8)); s5 = addU32(s5, rotr32(s7, GM_R8));
        s2 = (s2 ^ s4) >>> 0; s3 = (s3 ^ s5) >>> 0;
        s2 = rotr32(s2, GM_R4); s3 = rotr32(s3, GM_R4);
    } else {
        s0 = addU32(s0, rotr32(s2, HM_R5)); s1 = addU32(s1, rotr32(s3, HM_R5));
        s6 = (s6 ^ s0) >>> 0; s7 = (s7 ^ s1) >>> 0;
        s6 = rotr32(s6, HM_R1); s7 = rotr32(s7, HM_R1);
        s4 = addU32(s4, rotr32(s6, HM_R6)); s5 = addU32(s5, rotr32(s7, HM_R6));
        s2 = (s2 ^ s4) >>> 0; s3 = (s3 ^ s5) >>> 0;
        s2 = rotr32(s2, HM_R2); s3 = rotr32(s3, HM_R2);
        s0 = (s0 ^ rotr32(s4, HM_R9)) >>> 0; s1 = (s1 ^ rotr32(s5, HM_R9)) >>> 0;
        s0 = addU32(s0, rotr32(s2, HM_R7)); s1 = addU32(s1, rotr32(s3, HM_R7));
        s6 = (s6 ^ s0) >>> 0; s7 = (s7 ^ s1) >>> 0;
        s6 = rotr32(s6, HM_R3); s7 = rotr32(s7, HM_R3);
        s4 = addU32(s4, rotr32(s6, HM_R8)); s5 = addU32(s5, rotr32(s7, HM_R8));
        s2 = (s2 ^ s4) >>> 0; s3 = (s3 ^ s5) >>> 0;
        s2 = rotr32(s2, HM_R4); s3 = rotr32(s3, HM_R4);
    }
    // ShiftRows，单个 8-循环置换，循环 (0 2 7 1 5 4 3 6)
    let t = s0; s0 = s2; s2 = s7; s7 = s1;
    s1 = s5; s5 = s4; s4 = s3; s3 = s6; s6 = t;

    // 写回工作数组
    s[0] = s0; s[1] = s1; s[2] = s2; s[3] = s3;
    s[4] = s4; s[5] = s5; s[6] = s6; s[7] = s7;
}

// 压缩函数的工作数组，模块级复用，避免每块新建
const arxWork = new Uint32Array(8);

// 对一个 32 字节消息块做 16 轮 ARX 压缩，Davies-Meyer 前馈；last=true 时注入封口常数
function arxCompress(state, block, last) {
    // 消息字读成局部变量，省掉每块的数组分配
    const m0 = readU32LE(block, 0), m1 = readU32LE(block, 4);
    const m2 = readU32LE(block, 8), m3 = readU32LE(block, 12);
    const m4 = readU32LE(block, 16), m5 = readU32LE(block, 20);
    const m6 = readU32LE(block, 24), m7 = readU32LE(block, 28);

    // 保存旧状态，最后前馈要用
    const o0 = state[0], o1 = state[1], o2 = state[2], o3 = state[3];
    const o4 = state[4], o5 = state[5], o6 = state[6], o7 = state[7];

    // 封口常数只加在吸收阶段，不参与末尾前馈
    const seal = last ? finalConst : zeroSeal;

    // 消息模加注入（末块含封口常数）
    const s = arxWork;
    s[0] = addU32(addU32(o0, m0), seal[0]); s[1] = addU32(addU32(o1, m1), seal[1]);
    s[2] = addU32(addU32(o2, m2), seal[2]); s[3] = addU32(addU32(o3, m3), seal[3]);
    s[4] = addU32(addU32(o4, m4), seal[4]); s[5] = addU32(addU32(o5, m5), seal[5]);
    s[6] = addU32(addU32(o6, m6), seal[6]); s[7] = addU32(addU32(o7, m7), seal[7]);

    for (let r = 0; r < ARX_ROUNDS; r++) {
        arxRound(s, r);
        // 每轮异或圆周率派生的轮常数做域分离
        const c = roundConstants[r];
        s[0] = (s[0] ^ c[0]) >>> 0; s[1] = (s[1] ^ c[1]) >>> 0;
        s[2] = (s[2] ^ c[2]) >>> 0; s[3] = (s[3] ^ c[3]) >>> 0;
        s[4] = (s[4] ^ c[4]) >>> 0; s[5] = (s[5] ^ c[5]) >>> 0;
        s[6] = (s[6] ^ c[6]) >>> 0; s[7] = (s[7] ^ c[7]) >>> 0;
    }

    // Davies-Meyer 前馈
    state[0] = addU32((s[0] ^ o0) >>> 0, m0); state[1] = addU32((s[1] ^ o1) >>> 0, m1);
    state[2] = addU32((s[2] ^ o2) >>> 0, m2); state[3] = addU32((s[3] ^ o3) >>> 0, m3);
    state[4] = addU32((s[4] ^ o4) >>> 0, m4); state[5] = addU32((s[5] ^ o5) >>> 0, m5);
    state[6] = addU32((s[6] ^ o6) >>> 0, m6); state[7] = addU32((s[7] ^ o7) >>> 0, m7);
}

// 哈希

// 流式哈希
class Hash {
    constructor() {
        this.state = new Uint32Array(ARX_INIT_STATE);
        this.buf = new Uint8Array(32);
        this.bufLen = 0;
        this.total = 0n;
    }

    write(data) {
        data = toBytesStrict(data, '哈希输入');
        const n = data.length;
        this.total += BigInt(n);

        // 先把缓冲区填满
        if (this.bufLen > 0) {
            let copied = 32 - this.bufLen;
            if (copied > n) copied = n;
            this.buf.set(data.subarray(0, copied), this.bufLen);
            this.bufLen += copied;
            data = data.subarray(copied);
            if (this.bufLen < 32) return this;
            arxCompress(this.state, this.buf, false);
            this.bufLen = 0;
        }

        // 整块直接压缩
        let off = 0;
        while (off + 32 <= data.length) {
            arxCompress(this.state, data.subarray(off, off + 32), false);
            off += 32;
        }

        // 尾部存到缓冲区
        if (off < data.length) {
            this.buf.set(data.subarray(off), 0);
            this.bufLen = data.length - off;
        }
        return this;
    }

    // 算出最终摘要，长度域是小端的，跟标准 SHA-256 不一样
    sum() {
        const state = new Uint32Array(this.state);

        const dataLen = this.bufLen;
        let padNeed = 32 - (dataLen % 32);
        if (padNeed < 9) padNeed += 32;  // 至少要放 1 字节 0x80 加 8 字节长度

        const padded = new Uint8Array(64);  // 最多 64 字节就够
        padded.set(this.buf.subarray(0, dataLen), 0);
        padded[dataLen] = 0x80;
        // 比特长度，小端存放
        writeU64LE(padded, dataLen + padNeed - 8, this.total * 8n);

        const totalBlockBytes = dataLen + padNeed;
        for (let i = 0; i < totalBlockBytes; i += 32) {
            // 最后一个填充块带封口标志，标记这是消息末尾
            const last = i + 32 >= totalBlockBytes;
            arxCompress(state, padded.subarray(i, i + 32), last);
        }

        const out = new Uint8Array(32);
        for (let i = 0; i < 8; i++) writeU32LE(out, i * 4, state[i]);
        return out;
    }
}

// 一次性哈希
function hashSum(data) {
    const h = new Hash();
    h.write(data);
    return h.sum();
}

// HMAC

// HMAC 用两把带密钥的 Hash：inner 把 ipad 异或到初始状态，outer 把 opad 异或到初始状态
function hmacCompute(key, data) {
    // 字符串参数按 UTF-8 编码成字节
    key = toBytesStrict(key, 'HMAC 密钥');
    data = toBytesStrict(data, 'HMAC 数据');
    const k = new Uint8Array(32);
    if (key.length > 32) {
        // 长密钥先哈希
        const hashed = hashSum(key);
        k.set(hashed);
    } else {
        k.set(key);
    }

    const ipad = new Uint8Array(32), opad = new Uint8Array(32);
    for (let i = 0; i < 32; i++) {
        ipad[i] = k[i] ^ 0x36;
        opad[i] = k[i] ^ 0x5C;
    }

    // inner：ipad 异或到初始状态，再写入 data
    const innerH = new Hash();
    for (let i = 0; i < 8; i++) innerH.state[i] = (innerH.state[i] ^ readU32LE(ipad, i * 4)) >>> 0;
    innerH.write(data);
    const innerHash = innerH.sum();

    // outer：opad 异或到初始状态，再写入 innerHash
    const outerH = new Hash();
    for (let i = 0; i < 8; i++) outerH.state[i] = (outerH.state[i] ^ readU32LE(opad, i * 4)) >>> 0;
    outerH.write(innerHash);
    const result = outerH.sum();

    // 归一化后的密钥和两把填充密钥用完就抹掉
    wipe(k, ipad, opad);
    return result;
}

// HKDF

// HKDF 提取阶段：PRK = HMAC(salt, ikm)
function hkdfExtract(ikm, salt) {
    ikm = toBytesStrict(ikm, 'IKM');
    salt = toBytes(salt);
    if (!salt || salt.length === 0) salt = new Uint8Array(32);
    else salt = toBytesStrict(salt, 'salt');
    return hmacCompute(salt, ikm);
}

// HKDF 扩展阶段，输出指定长度
function hkdfExpand(prk, info, length) {
    prk = toBytesStrict(prk, 'PRK');
    info = info === undefined || info === null ? new Uint8Array(0) : toBytesStrict(info, 'info');
    // 上限 255 块：块计数器是单字节，超了会重复派生；同时挡住 NaN/小数，避免静默返回空密钥
    requireSafeInt(length, 'HKDF 输出长度', 1, 255 * 32);

    const result = new Uint8Array(length);
    let prev = new Uint8Array(0);
    let offset = 0;
    let counter = 1;

    while (offset < length) {
        // T(i) = HMAC(prk, T(i-1) || info || counter)
        const input = new Uint8Array(prev.length + info.length + 1);
        input.set(prev, 0);
        input.set(info, prev.length);
        input[prev.length + info.length] = counter;
        prev = hmacCompute(prk, input);

        const copyLen = Math.min(32, length - offset);
        result.set(prev.subarray(0, copyLen), offset);
        offset += 32;
        counter++;
        wipe(input);
    }

    wipe(prev);
    return result;
}

// 完整 HKDF
function hkdf(ikm, salt, info, length) {
    const prk = hkdfExtract(ikm, salt);
    return hkdfExpand(prk, info, length);
}

// PBKDF2

// PBKDF2 密码基密钥派生，块索引按 RFC 2898 用大端序
function pbkdf2(password, salt, iterations, keyLen) {
    // 字符串参数按 UTF-8 编码成字节
    password = toBytesStrict(password, '口令');
    salt = toBytes(salt);
    if (salt === undefined || salt === null) salt = new Uint8Array(0);
    else salt = toBytesStrict(salt, 'salt');
    // 迭代次数必须是有限整数：NaN 会退化成只迭代一次，Infinity 会让循环永不结束
    requireSafeInt(iterations, 'PBKDF2 迭代次数', 1, 0x7fffffff);
    // 输出长度上限防超大 keyLen 直接把内存吃爆
    requireSafeInt(keyLen, 'PBKDF2 输出长度', 1, 0x7ffffff0);
    if (password.length === 0) throw new Error('PBKDF2: 口令不能为空');

    const numBlocks = Math.ceil(keyLen / 32);
    const result = new Uint8Array(numBlocks * 32);

    for (let blockIdx = 1; blockIdx <= numBlocks; blockIdx++) {
        // U1 = HMAC(password, salt || BE(blockIdx))
        const input = new Uint8Array(salt.length + 4);
        input.set(salt, 0);
        input[salt.length] = (blockIdx >>> 24) & 0xff;
        input[salt.length + 1] = (blockIdx >>> 16) & 0xff;
        input[salt.length + 2] = (blockIdx >>> 8) & 0xff;
        input[salt.length + 3] = blockIdx & 0xff;

        let uPrev = hmacCompute(password, input);
        const accumulator = new Uint8Array(uPrev);  // 拷贝一份

        for (let j = 2; j <= iterations; j++) {
            const uCurr = hmacCompute(password, uPrev);
            for (let k = 0; k < 32; k++) accumulator[k] ^= uCurr[k];
            wipe(uPrev);
            uPrev = uCurr;
        }
        wipe(uPrev, input);
        result.set(accumulator, (blockIdx - 1) * 32);
        wipe(accumulator);
    }

    return result.subarray(0, keyLen);
}

// DeriveKey

// BLAKE3 风格的派生：context + 0x00 + keyMaterial 哈希后再 CTR 扩展
function deriveKey(context, keyMaterial, length) {
    // NaN/小数长度会静默返回空数组，这里一律拒绝
    requireSafeInt(length, 'DeriveKey 输出长度', 0, 0x7fffffff);
    if (length === 0) return new Uint8Array(0);

    // 构建 context + 0x00 + keyMaterial
    const ctxBytes = toBytesStrict(context, 'context');
    keyMaterial = toBytesStrict(keyMaterial, '密钥材料');
    const input = new Uint8Array(ctxBytes.length + 1 + keyMaterial.length);
    input.set(ctxBytes, 0);
    input[ctxBytes.length] = 0x00;
    input.set(keyMaterial, ctxBytes.length + 1);

    // 计算基础哈希作为派生密钥的根密钥
    const baseKey = hashSum(input);
    wipe(input);

    if (length <= 32) return baseKey.subarray(0, length);

    // 超过 32 字节用 CTR 扩展，零nonce
    const nonce = new Uint8Array(NONCE_SIZE);
    const zeros = new Uint8Array(length);
    const out = ctrCrypt(baseKey, nonce, zeros);
    wipe(baseKey);
    return out;
}

// CTR 模式

// CTR 批量处理，nonce 为 24 字节，内部计数器从 0 开始递增，加解密同一个操作
function ctrCrypt(key, nonce, data) {
    // 校验放在空数据早退之前：否则非法密钥/nonce 会被"空输入"放过，各端行为不一致
    key = toBytesStrict(key, '密钥');
    nonce = toBytesStrict(nonce, 'nonce');
    data = toBytesStrict(data, '数据');
    if (nonce.length !== NONCE_SIZE) throw new Error(`CTR: nonce 必须 ${NONCE_SIZE} 字节`);
    if (data.length === 0) return new Uint8Array(0);
    const c = new BlockCipher(key);
    const out = new Uint8Array(data.length);

    // 内部构造32字节counter块：前24字节nonce + 后8字节计数器初值0
    const ctrBuf = new Uint8Array(32);
    ctrBuf.set(nonce, 0);
    // 计数器拆成两个 32 位半字，低位溢出进位到高位，省掉内层的 BigInt 运算
    let lo = 0, hi = 0;

    let offset = 0;
    const ks = new Uint8Array(32);
    while (offset + 32 <= data.length) {
        writeU32LE(ctrBuf, 24, lo);
        writeU32LE(ctrBuf, 28, hi);
        c.encryptBlock(ctrBuf, ks);
        for (let i = 0; i < 32; i++) out[offset + i] = data[offset + i] ^ ks[i];
        lo = (lo + 1) >>> 0;
        if (lo === 0) hi = (hi + 1) >>> 0;
        offset += 32;
    }
    // 尾部不足一块
    if (offset < data.length) {
        writeU32LE(ctrBuf, 24, lo);
        writeU32LE(ctrBuf, 28, hi);
        c.encryptBlock(ctrBuf, ks);
        const remain = data.length - offset;
        for (let i = 0; i < remain; i++) out[offset + i] = data[offset + i] ^ ks[i];
    }

    c.zeroize();
    return out;
}

// CBC 模式

// CBC 加密，PKCS#7 填充
function cbcEncrypt(key, iv, plaintext) {
    key = toBytesStrict(key, '密钥');
    iv = toBytesStrict(iv, 'IV');
    plaintext = toBytesStrict(plaintext, '明文');
    // IV 必须是完整块长，短 IV 会在 prev 中静默补零，导致与其他端结果不一致
    if (iv.length !== BLOCK_SIZE) {
        throw new Error(`CBC: IV 必须 ${BLOCK_SIZE} 字节`);
    }
    const c = new BlockCipher(key);
    const out = new Uint8Array(plaintext.length + 32);  // 最坏情况多一个完整填充块
    const prev = new Uint8Array(32);
    prev.set(iv);

    const fullBlocks = Math.floor(plaintext.length / 32);
    const lastLen = plaintext.length - fullBlocks * 32;

    // 完整块
    const blk = new Uint8Array(32);
    for (let b = 0; b < fullBlocks; b++) {
        for (let i = 0; i < 32; i++) blk[i] = plaintext[b * 32 + i] ^ prev[i];
        c.encryptBlock(blk, out.subarray(b * 32, b * 32 + 32));
        prev.set(out.subarray(b * 32, b * 32 + 32));
    }

    // 最后一块带 PKCS#7 填充，空数据也要加一个完整块
    const padLen = 32 - lastLen;
    const last = new Uint8Array(32);
    last.set(plaintext.subarray(fullBlocks * 32), 0);
    for (let i = lastLen; i < 32; i++) last[i] = padLen;
    for (let i = 0; i < 32; i++) last[i] ^= prev[i];
    c.encryptBlock(last, out.subarray(fullBlocks * 32, fullBlocks * 32 + 32));

    c.zeroize();
    return out.subarray(0, (fullBlocks + 1) * 32);
}

// CBC 解密，常量时间校验填充
function cbcDecrypt(key, iv, ciphertext) {
    key = toBytesStrict(key, '密钥');
    iv = toBytesStrict(iv, 'IV');
    ciphertext = toBytesStrict(ciphertext, '密文');
    if (iv.length !== BLOCK_SIZE) {
        throw new Error(`CBC: IV 必须 ${BLOCK_SIZE} 字节`);
    }
    if (ciphertext.length === 0 || ciphertext.length % 32 !== 0) {
        throw new Error('CBC: 密文长度必须是 32 的整数倍');
    }
    const c = new BlockCipher(key);
    const out = new Uint8Array(ciphertext.length);
    const blocks = ciphertext.length / 32;

    // 逐块解密
    const prev = new Uint8Array(32);
    prev.set(iv);
    const dec = new Uint8Array(32);
    for (let b = 0; b < blocks; b++) {
        c.decryptBlock(ciphertext.subarray(b * 32, b * 32 + 32), dec);
        for (let i = 0; i < 32; i++) out[b * 32 + i] = dec[i] ^ prev[i];
        prev.set(ciphertext.subarray(b * 32, b * 32 + 32));
    }

    // 常量时间校验 PKCS#7 填充
    const lastOff = (blocks - 1) * 32;
    const padLen = out[lastOff + 31];
    const pad = padLen >>> 0;
    let bad = ((((pad - 1) >>> 31) | ((32 - pad) >>> 31)) & 1);
    for (let i = 0; i < 32; i++) {
        const b = out[lastOff + 31 - i];
        const idxDiff = (padLen - i - 1) | 0;  // 转成有符号 32 位
        const mask = (0 - (((idxDiff >>> 31) ^ 1) & 0xff)) & 0xff;
        bad |= ((b ^ padLen) & mask) & 0xff;
    }
    if (bad !== 0) {
        c.zeroize();
        // 填充不对说明密文被改动过，解出来的明文不可信，别留在调用方能拿到的内存里
        out.fill(0);
        throw new Error('CBC: 填充不对');
    }

    c.zeroize();
    return out.subarray(0, ciphertext.length - padLen);
}

// XTS 模式

// GF(2^256) 乘 2，多项式 0x0425，进位时异或到倒数第二字节和最后一字节
function gfMul2_256(x) {
    const carry = x[0] >>> 7;
    for (let i = 0; i < 31; i++) {
        x[i] = ((x[i] << 1) | (x[i + 1] >>> 7)) & 0xff;
    }
    x[31] = (x[31] << 1) & 0xff;
    if (carry) {
        x[30] ^= 0x04;
        x[31] ^= 0x25;
    }
}

// XTS 处理，支持密文窃取
function xtsProcess(c1, c2, data, sector, encrypt) {
    const out = new Uint8Array(data.length);
    out.set(data);

    // tweak = E_key2(sector 小端填充 32 字节)
    const tweak = new Uint8Array(32);
    writeU64LE(tweak, 24, BigInt(sector));
    c2.encryptBlock(tweak, tweak);

    const fullBlocks = Math.floor(data.length / 32);
    const partial = data.length % 32;
    let processBlocks = fullBlocks;
    if (partial > 0) processBlocks = fullBlocks - 1;  // 留一块做密文窃取

    // 完整块
    const blk = new Uint8Array(32);
    for (let b = 0; b < processBlocks; b++) {
        const off = b * 32;
        for (let i = 0; i < 32; i++) blk[i] = out[off + i] ^ tweak[i];
        if (encrypt) c1.encryptBlock(blk, blk);
        else c1.decryptBlock(blk, blk);
        for (let i = 0; i < 32; i++) out[off + i] = blk[i] ^ tweak[i];
        gfMul2_256(tweak);
    }

    // 密文窃取处理最后的不完整块
    if (partial > 0 && fullBlocks >= 1) {
        const prevOff = (fullBlocks - 1) * 32;
        const lastOff = fullBlocks * 32;

        const Ppartial = new Uint8Array(32).fill(0);
        const Pprev = new Uint8Array(32);
        const CC = new Uint8Array(32);
        const PP = new Uint8Array(32);
        Ppartial.set(out.subarray(lastOff, lastOff + partial), 0);
        Pprev.set(out.subarray(prevOff, prevOff + 32));

        if (encrypt) {
            // 加密倒数第二块当 CC
            CC.set(Pprev);
            for (let i = 0; i < 32; i++) CC[i] ^= tweak[i];
            c1.encryptBlock(CC, CC);
            for (let i = 0; i < 32; i++) CC[i] ^= tweak[i];
            // CC 前 m 字节写到最后一块位置
            out.set(CC.subarray(0, partial), lastOff);
            // PP = CC 尾部 + Ppartial
            PP.set(CC.subarray(partial, 32), 0);
            PP.set(Ppartial.subarray(0, partial), 32 - partial);
            gfMul2_256(tweak);
            for (let i = 0; i < 32; i++) PP[i] ^= tweak[i];
            c1.encryptBlock(PP, PP);
            for (let i = 0; i < 32; i++) PP[i] ^= tweak[i];
            out.set(PP, prevOff);
        } else {
            // 解密：先算 tweakN1
            const tweakN1 = new Uint8Array(tweak);
            gfMul2_256(tweakN1);

            const Cpartial = new Uint8Array(32).fill(0);
            Cpartial.set(out.subarray(lastOff, lastOff + partial), 0);

            // 解密最后一整块（在 prevOff 位置）用 tweakN1
            PP.set(out.subarray(prevOff, prevOff + 32));
            for (let i = 0; i < 32; i++) PP[i] ^= tweakN1[i];
            c1.decryptBlock(PP, PP);
            for (let i = 0; i < 32; i++) PP[i] ^= tweakN1[i];
            // PP 尾部 m 字节是原明文尾部，写到最后一块
            out.set(PP.subarray(32 - partial, 32), lastOff);

            // CC = Cpartial + PP 前部，用 tweak 解密得倒数第二块明文
            CC.set(Cpartial.subarray(0, partial), 0);
            CC.set(PP.subarray(0, 32 - partial), partial);
            for (let i = 0; i < 32; i++) CC[i] ^= tweak[i];
            c1.decryptBlock(CC, CC);
            for (let i = 0; i < 32; i++) CC[i] ^= tweak[i];
            out.set(CC, prevOff);
        }
    }

    return out;
}

// XTS 扇区号是 uint64：负数和超范围值会被二进制补码悄悄回绕成另一个扇区号，直接拒绝
function requireSector(sector) {
    if (typeof sector === 'bigint') {
        if (sector < 0n || sector > 0xffffffffffffffffn) {
            throw new RangeError(`XTS: 扇区号超出 uint64 范围: ${sector}`);
        }
        return sector;
    }
    requireSafeInt(sector, 'XTS 扇区号', 0, Number.MAX_SAFE_INTEGER);
    return BigInt(sector);
}

// XTS 加密
function xtsEncrypt(key1, key2, sector, plaintext) {
    key1 = toBytesStrict(key1, '密钥1');
    key2 = toBytesStrict(key2, '密钥2');
    plaintext = toBytesStrict(plaintext, '明文');
    sector = requireSector(sector);
    if (plaintext.length < BLOCK_SIZE) {
        throw new Error(`XTS: 数据至少要 ${BLOCK_SIZE} 字节`);
    }
    const c1 = new BlockCipher(key1);
    const c2 = new BlockCipher(key2);
    const result = xtsProcess(c1, c2, plaintext, sector, true);
    c1.zeroize();
    c2.zeroize();
    return result;
}

// XTS 解密
function xtsDecrypt(key1, key2, sector, ciphertext) {
    key1 = toBytesStrict(key1, '密钥1');
    key2 = toBytesStrict(key2, '密钥2');
    ciphertext = toBytesStrict(ciphertext, '密文');
    sector = requireSector(sector);
    if (ciphertext.length < BLOCK_SIZE) {
        throw new Error(`XTS: 数据至少要 ${BLOCK_SIZE} 字节`);
    }
    const c1 = new BlockCipher(key1);
    const c2 = new BlockCipher(key2);
    const result = xtsProcess(c1, c2, ciphertext, sector, false);
    c1.zeroize();
    c2.zeroize();
    return result;
}

// AEAD

// 从主密钥和 nonce 派生加密密钥和认证密钥，nonce 当 HKDF 的 salt
function deriveAEADKeys(master, nonce) {
    const prk = hkdfExtract(master, nonce);
    const encKey = hkdfExpand(prk, new TextEncoder().encode('aead-key'), 32);
    const macKey = hkdfExpand(prk, new TextEncoder().encode('tag-key'), 32);
    return { encKey, macKey };
}

// 计算认证标签：HMAC(macKey, aad || ct || le64(aadLen) || le64(ctLen))，取前 16 字节
function aeadComputeTag(macKey, aad, ct) {
    const aadLen = aad.length;
    const ctLen = ct.length;
    const buf = new Uint8Array(aadLen + ctLen + 16);
    buf.set(aad, 0);
    buf.set(ct, aadLen);
    writeU64LE(buf, aadLen + ctLen, BigInt(aadLen));
    writeU64LE(buf, aadLen + ctLen + 8, BigInt(ctLen));
    const fullTag = hmacCompute(macKey, buf);
    return fullTag.subarray(0, 16);
}

// AEAD 加密，输出 ciphertext || tag
function aeadEncrypt(key, nonce, plaintext, ad) {
    key = toBytesStrict(key, '密钥');
    nonce = toBytesStrict(nonce, 'nonce');
    plaintext = toBytesStrict(plaintext, '明文');
    if (key.length !== KEY_SIZE) throw new Error(`AEAD: 密钥必须 ${KEY_SIZE} 字节`);
    if (nonce.length !== NONCE_SIZE) throw new Error(`AEAD: nonce 必须 ${NONCE_SIZE} 字节`);
    ad = ad === undefined || ad === null ? new Uint8Array(0) : toBytesStrict(ad, '附加数据');

    const { encKey, macKey } = deriveAEADKeys(key, nonce);
    try {
        // CTR 加密，nonce 当计数器块前 24 字节，后 8 字节从 0 开始
        const ciphertext = ctrCrypt(encKey, nonce, plaintext);

        // 算认证标签
        const tag = aeadComputeTag(macKey, ad, ciphertext);

        const out = new Uint8Array(ciphertext.length + 16);
        out.set(ciphertext, 0);
        out.set(tag, ciphertext.length);
        return out;
    } finally {
        // 子密钥是一次性的，出函数就抹掉
        wipe(encKey, macKey);
    }
}

// AEAD 解密，输入是 ciphertext || tag，验证不过抛异常
function aeadDecrypt(key, nonce, ciphertextWithTag, ad) {
    key = toBytesStrict(key, '密钥');
    nonce = toBytesStrict(nonce, 'nonce');
    ciphertextWithTag = toBytesStrict(ciphertextWithTag, '密文');
    if (key.length !== KEY_SIZE) throw new Error(`AEAD: 密钥必须 ${KEY_SIZE} 字节`);
    if (nonce.length !== NONCE_SIZE) throw new Error(`AEAD: nonce 必须 ${NONCE_SIZE} 字节`);
    if (ciphertextWithTag.length < 16) throw new Error('AEAD: 数据太短');
    ad = ad === undefined || ad === null ? new Uint8Array(0) : toBytesStrict(ad, '附加数据');

    const ctLen = ciphertextWithTag.length - 16;
    const ct = ciphertextWithTag.subarray(0, ctLen);
    const expectedTag = ciphertextWithTag.subarray(ctLen);

    const { encKey, macKey } = deriveAEADKeys(key, nonce);
    try {
        // 先验证标签，过了再解密
        const actualTag = aeadComputeTag(macKey, ad, ct);
        if (!constantTimeCompare(actualTag, expectedTag)) {
            throw new Error('AEAD: 认证失败');
        }

        // CTR 解密
        return ctrCrypt(encKey, nonce, ct);
    } finally {
        wipe(encKey, macKey);
    }
}

// XOF

// 可变长度输出 XOF，先算哈希，再用哈希值当密钥对零块做 CTR 加密
function xof(input, length) {
    // NaN/小数长度会静默返回空数组，这里统一要求有界安全整数
    requireSafeInt(length, 'XOF 输出长度', 0, 0x7fffffff);
    if (length === 0) return new Uint8Array(0);
    input = toBytesStrict(input, '输入');
    const state = hashSum(input);
    const c = new BlockCipher(state);

    const counter = new Uint8Array(32);
    const out = new Uint8Array(length);
    const ks = new Uint8Array(32);
    let baseCtr = 0n;
    let written = 0;
    while (written < length) {
        const end = Math.min(written + 32, length);
        writeU64LE(counter, 24, baseCtr);
        c.encryptBlock(counter, ks);
        const n = end - written;
        for (let i = 0; i < n; i++) out[written + i] = ks[i];
        written = end;
        baseCtr++;
    }
    c.zeroize();
    return out;
}

// 单块加解密对外接口

function blockEncrypt(key, plaintext) {
    key = toBytesStrict(key, '密钥');
    plaintext = toBytesStrict(plaintext, '明文');
    if (plaintext.length !== BLOCK_SIZE) throw new Error(`明文必须 ${BLOCK_SIZE} 字节`);
    const c = new BlockCipher(key);
    const out = new Uint8Array(32);
    c.encryptBlock(plaintext, out);
    c.zeroize();
    return out;
}

function blockDecrypt(key, ciphertext) {
    key = toBytesStrict(key, '密钥');
    ciphertext = toBytesStrict(ciphertext, '密文');
    if (ciphertext.length !== BLOCK_SIZE) throw new Error(`密文必须 ${BLOCK_SIZE} 字节`);
    const c = new BlockCipher(key);
    const out = new Uint8Array(32);
    c.decryptBlock(ciphertext, out);
    c.zeroize();
    return out;
}

// 导出接口

return {
    // 常量
    BLOCK_SIZE,
    KEY_SIZE,
    NONCE_SIZE,
    TAG_SIZE,
    ROUNDS,

    // 分组密码
    BlockCipher,
    blockEncrypt,
    blockDecrypt,

    // CTR
    ctrCrypt,

    // CBC
    cbcEncrypt,
    cbcDecrypt,

    // XTS
    xtsEncrypt,
    xtsDecrypt,

    // AEAD
    aeadEncrypt,
    aeadDecrypt,

    // 哈希
    Hash,
    hash: hashSum,
    hashSum,
    xof,

    // HMAC
    hmac: hmacCompute,

    // KDF
    hkdf,
    pbkdf2,
    deriveKey,

    // 工具
    constantTimeCompare,
};

}));
