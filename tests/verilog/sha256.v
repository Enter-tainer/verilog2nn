// T13: Full SHA-256 single block - combinational logic, all 64 steps unrolled
// Input: 512-bit message block + 256-bit initial hash (h0..h7)
// Output: 256-bit hash
// Total input: 768 bits, output: 256 bits

module sha256(
    input [511:0] msg,
    input [31:0] h0_in,
    input [31:0] h1_in,
    input [31:0] h2_in,
    input [31:0] h3_in,
    input [31:0] h4_in,
    input [31:0] h5_in,
    input [31:0] h6_in,
    input [31:0] h7_in,
    output [31:0] h0_out,
    output [31:0] h1_out,
    output [31:0] h2_out,
    output [31:0] h3_out,
    output [31:0] h4_out,
    output [31:0] h5_out,
    output [31:0] h6_out,
    output [31:0] h7_out
);

    // SHA-256 round constants
    wire [31:0] K [0:63];
    assign K[0]  = 32'h428a2f98; assign K[1]  = 32'h71374491;
    assign K[2]  = 32'hb5c0fbcf; assign K[3]  = 32'he9b5dba5;
    assign K[4]  = 32'h3956c25b; assign K[5]  = 32'h59f111f1;
    assign K[6]  = 32'h923f82a4; assign K[7]  = 32'hab1c5ed5;
    assign K[8]  = 32'hd807aa98; assign K[9]  = 32'h12835b01;
    assign K[10] = 32'h243185be; assign K[11] = 32'h550c7dc3;
    assign K[12] = 32'h72be5d74; assign K[13] = 32'h80deb1fe;
    assign K[14] = 32'h9bdc06a7; assign K[15] = 32'hc19bf174;
    assign K[16] = 32'he49b69c1; assign K[17] = 32'hefbe4786;
    assign K[18] = 32'h0fc19dc6; assign K[19] = 32'h240ca1cc;
    assign K[20] = 32'h2de92c6f; assign K[21] = 32'h4a7484aa;
    assign K[22] = 32'h5cb0a9dc; assign K[23] = 32'h76f988da;
    assign K[24] = 32'h983e5152; assign K[25] = 32'ha831c66d;
    assign K[26] = 32'hb00327c8; assign K[27] = 32'hbf597fc7;
    assign K[28] = 32'hc6e00bf3; assign K[29] = 32'hd5a79147;
    assign K[30] = 32'h06ca6351; assign K[31] = 32'h14292967;
    assign K[32] = 32'h27b70a85; assign K[33] = 32'h2e1b2138;
    assign K[34] = 32'h4d2c6dfc; assign K[35] = 32'h53380d13;
    assign K[36] = 32'h650a7354; assign K[37] = 32'h766a0abb;
    assign K[38] = 32'h81c2c92e; assign K[39] = 32'h92722c85;
    assign K[40] = 32'ha2bfe8a1; assign K[41] = 32'ha81a664b;
    assign K[42] = 32'hc24b8b70; assign K[43] = 32'hc76c51a3;
    assign K[44] = 32'hd192e819; assign K[45] = 32'hd6990624;
    assign K[46] = 32'hf40e3585; assign K[47] = 32'h106aa070;
    assign K[48] = 32'h19a4c116; assign K[49] = 32'h1e376c08;
    assign K[50] = 32'h2748774c; assign K[51] = 32'h34b0bcb5;
    assign K[52] = 32'h391c0cb3; assign K[53] = 32'h4ed8aa4a;
    assign K[54] = 32'h5b9cca4f; assign K[55] = 32'h682e6ff3;
    assign K[56] = 32'h748f82ee; assign K[57] = 32'h78a5636f;
    assign K[58] = 32'h84c87814; assign K[59] = 32'h8cc70208;
    assign K[60] = 32'h90befffa; assign K[61] = 32'ha4506ceb;
    assign K[62] = 32'hbef9a3f7; assign K[63] = 32'hc67178f2;

    // Message schedule W[0..63]
    // W[0..15] = message words (big-endian)
    wire [31:0] W [0:63];
    genvar gi;
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : msg_words
            assign W[gi] = msg[(15-gi)*32 +: 32];
        end
    endgenerate

    // W[16..63]: sigma0 and sigma1 expansion
    // sigma0(x) = ROTR(7,x) ^ ROTR(18,x) ^ SHR(3,x)
    // sigma1(x) = ROTR(17,x) ^ ROTR(19,x) ^ SHR(10,x)
    // W[i] = sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16]

    // Helper functions as wire expressions
    `define SIGMA0(x) ({x[6:0], x[31:7]} ^ {x[17:0], x[31:18]} ^ {3'b000, x[31:3]})
    `define SIGMA1(x) ({x[16:0], x[31:17]} ^ {x[18:0], x[31:19]} ^ {10'b0, x[31:10]})

    generate
        for (gi = 16; gi < 64; gi = gi + 1) begin : w_expand
            wire [31:0] s0 = `SIGMA0(W[gi-15]);
            wire [31:0] s1 = `SIGMA1(W[gi-2]);
            assign W[gi] = s1 + W[gi-7] + s0 + W[gi-16];
        end
    endgenerate

    // Compression function - 64 rounds
    // Working variables: a, b, c, d, e, f, g, h
    // Each round:
    //   Sigma1(e) = ROTR(6,e) ^ ROTR(11,e) ^ ROTR(25,e)
    //   Ch(e,f,g) = (e & f) ^ (~e & g)
    //   temp1 = h + Sigma1(e) + Ch(e,f,g) + K[i] + W[i]
    //   Sigma0(a) = ROTR(2,a) ^ ROTR(13,a) ^ ROTR(22,a)
    //   Maj(a,b,c) = (a & b) ^ (a & c) ^ (b & c)
    //   temp2 = Sigma0(a) + Maj(a,b,c)
    //   h=g, g=f, f=e, e=d+temp1, d=c, c=b, b=a, a=temp1+temp2

    wire [31:0] sa [0:64];
    wire [31:0] sb [0:64];
    wire [31:0] scc [0:64]; // 'sc' conflicts with genvar scope in some tools
    wire [31:0] sd [0:64];
    wire [31:0] se [0:64];
    wire [31:0] sf [0:64];
    wire [31:0] sg [0:64];
    wire [31:0] sh [0:64];

    assign sa[0] = h0_in;
    assign sb[0] = h1_in;
    assign scc[0] = h2_in;
    assign sd[0] = h3_in;
    assign se[0] = h4_in;
    assign sf[0] = h5_in;
    assign sg[0] = h6_in;
    assign sh[0] = h7_in;

    // Big Sigma functions for compression
    `define BIGSIGMA0(x) ({x[1:0], x[31:2]} ^ {x[12:0], x[31:13]} ^ {x[21:0], x[31:22]})
    `define BIGSIGMA1(x) ({x[5:0], x[31:6]} ^ {x[10:0], x[31:11]} ^ {x[24:0], x[31:25]})

    // Unroll all 64 rounds using generate
    wire [31:0] ch_val [0:63];
    wire [31:0] maj_val [0:63];
    wire [31:0] sig0_val [0:63];
    wire [31:0] sig1_val [0:63];
    wire [31:0] temp1 [0:63];
    wire [31:0] temp2 [0:63];

    generate
        for (gi = 0; gi < 64; gi = gi + 1) begin : rounds
            assign sig1_val[gi] = `BIGSIGMA1(se[gi]);
            assign ch_val[gi] = (se[gi] & sf[gi]) ^ (~se[gi] & sg[gi]);
            assign temp1[gi] = sh[gi] + sig1_val[gi] + ch_val[gi] + K[gi] + W[gi];

            assign sig0_val[gi] = `BIGSIGMA0(sa[gi]);
            assign maj_val[gi] = (sa[gi] & sb[gi]) ^ (sa[gi] & scc[gi]) ^ (sb[gi] & scc[gi]);
            assign temp2[gi] = sig0_val[gi] + maj_val[gi];

            assign sh[gi+1] = sg[gi];
            assign sg[gi+1] = sf[gi];
            assign sf[gi+1] = se[gi];
            assign se[gi+1] = sd[gi] + temp1[gi];
            assign sd[gi+1] = scc[gi];
            assign scc[gi+1] = sb[gi];
            assign sb[gi+1] = sa[gi];
            assign sa[gi+1] = temp1[gi] + temp2[gi];
        end
    endgenerate

    // Final addition
    assign h0_out = h0_in + sa[64];
    assign h1_out = h1_in + sb[64];
    assign h2_out = h2_in + scc[64];
    assign h3_out = h3_in + sd[64];
    assign h4_out = h4_in + se[64];
    assign h5_out = h5_in + sf[64];
    assign h6_out = h6_in + sg[64];
    assign h7_out = h7_in + sh[64];

endmodule
