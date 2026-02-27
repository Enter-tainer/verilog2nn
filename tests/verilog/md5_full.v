// T12: Full MD5 hash - combinational logic, all 4 rounds (64 steps) unrolled
// Input: 512-bit message block + 128-bit initial state (a0,b0,c0,d0)
// Output: 128-bit hash (a_out,b_out,c_out,d_out)
// Total input: 640 bits, output: 128 bits

module md5_full(
    input [511:0] msg,
    input [31:0] a0,
    input [31:0] b0,
    input [31:0] c0,
    input [31:0] d0,
    output [31:0] a_out,
    output [31:0] b_out,
    output [31:0] c_out,
    output [31:0] d_out
);

    // Extract 16 message words (little-endian 32-bit words)
    wire [31:0] M [0:15];
    genvar gi;
    generate
        for (gi = 0; gi < 16; gi = gi + 1) begin : msg_extract
            assign M[gi] = msg[gi*32 +: 32];
        end
    endgenerate

    // MD5 round constants T[i] = floor(2^32 * abs(sin(i+1)))
    // All 64 constants
    wire [31:0] K [0:63];
    assign K[0]  = 32'hd76aa478; assign K[1]  = 32'he8c7b756;
    assign K[2]  = 32'h242070db; assign K[3]  = 32'hc1bdceee;
    assign K[4]  = 32'hf57c0faf; assign K[5]  = 32'h4787c62a;
    assign K[6]  = 32'ha8304613; assign K[7]  = 32'hfd469501;
    assign K[8]  = 32'h698098d8; assign K[9]  = 32'h8b44f7af;
    assign K[10] = 32'hffff5bb1; assign K[11] = 32'h895cd7be;
    assign K[12] = 32'h6b901122; assign K[13] = 32'hfd987193;
    assign K[14] = 32'ha679438e; assign K[15] = 32'h49b40821;
    assign K[16] = 32'hf61e2562; assign K[17] = 32'hc040b340;
    assign K[18] = 32'h265e5a51; assign K[19] = 32'he9b6c7aa;
    assign K[20] = 32'hd62f105d; assign K[21] = 32'h02441453;
    assign K[22] = 32'hd8a1e681; assign K[23] = 32'he7d3fbc8;
    assign K[24] = 32'h21e1cde6; assign K[25] = 32'hc33707d6;
    assign K[26] = 32'hf4d50d87; assign K[27] = 32'h455a14ed;
    assign K[28] = 32'ha9e3e905; assign K[29] = 32'hfcefa3f8;
    assign K[30] = 32'h676f02d9; assign K[31] = 32'h8d2a4c8a;
    assign K[32] = 32'hfffa3942; assign K[33] = 32'h8771f681;
    assign K[34] = 32'h6d9d6122; assign K[35] = 32'hfde5380c;
    assign K[36] = 32'ha4beea44; assign K[37] = 32'h4bdecfa9;
    assign K[38] = 32'hf6bb4b60; assign K[39] = 32'hbebfbc70;
    assign K[40] = 32'h289b7ec6; assign K[41] = 32'heaa127fa;
    assign K[42] = 32'hd4ef3085; assign K[43] = 32'h04881d05;
    assign K[44] = 32'hd9d4d039; assign K[45] = 32'he6db99e5;
    assign K[46] = 32'h1fa27cf8; assign K[47] = 32'hc4ac5665;
    assign K[48] = 32'hf4292244; assign K[49] = 32'h432aff97;
    assign K[50] = 32'hab9423a7; assign K[51] = 32'hfc93a039;
    assign K[52] = 32'h655b59c3; assign K[53] = 32'h8f0ccc92;
    assign K[54] = 32'hffeff47d; assign K[55] = 32'h85845dd1;
    assign K[56] = 32'h6fa87e4f; assign K[57] = 32'hfe2ce6e0;
    assign K[58] = 32'ha3014314; assign K[59] = 32'h4e0811a1;
    assign K[60] = 32'hf7537e82; assign K[61] = 32'hbd3af235;
    assign K[62] = 32'h2ad7d2bb; assign K[63] = 32'heb86d391;

    // Per-round shift amounts
    // Round 1: 7, 12, 17, 22 (repeated 4 times)
    // Round 2: 5, 9, 14, 20 (repeated 4 times)
    // Round 3: 4, 11, 16, 23 (repeated 4 times)
    // Round 4: 6, 10, 15, 21 (repeated 4 times)

    // Message schedule index g for each step
    // Round 1: g = i
    // Round 2: g = (5*i + 1) mod 16
    // Round 3: g = (3*i + 5) mod 16
    // Round 4: g = (7*i) mod 16

    // Intermediate state wires: 65 sets of (a, b, c, d)
    wire [31:0] sa [0:64];
    wire [31:0] sb [0:64];
    wire [31:0] sc [0:64];
    wire [31:0] sd [0:64];

    assign sa[0] = a0;
    assign sb[0] = b0;
    assign sc[0] = c0;
    assign sd[0] = d0;

    // Function wires and step computation for each of 64 steps
    wire [31:0] f_val [0:63];
    wire [31:0] step_sum [0:63];
    wire [31:0] rotated [0:63];

    // Round 1 (steps 0-15): F(B,C,D) = (B & C) | (~B & D), g = i
    // Shift amounts: 7, 12, 17, 22, 7, 12, 17, 22, ...

    // Step 0
    assign f_val[0] = (sb[0] & sc[0]) | (~sb[0] & sd[0]);
    assign step_sum[0] = sa[0] + f_val[0] + K[0] + M[0];
    assign rotated[0] = {step_sum[0][24:0], step_sum[0][31:25]}; // <<< 7
    assign sa[1] = sd[0]; assign sb[1] = sb[0] + rotated[0]; assign sc[1] = sb[0]; assign sd[1] = sc[0];

    // Step 1
    assign f_val[1] = (sb[1] & sc[1]) | (~sb[1] & sd[1]);
    assign step_sum[1] = sa[1] + f_val[1] + K[1] + M[1];
    assign rotated[1] = {step_sum[1][19:0], step_sum[1][31:20]}; // <<< 12
    assign sa[2] = sd[1]; assign sb[2] = sb[1] + rotated[1]; assign sc[2] = sb[1]; assign sd[2] = sc[1];

    // Step 2
    assign f_val[2] = (sb[2] & sc[2]) | (~sb[2] & sd[2]);
    assign step_sum[2] = sa[2] + f_val[2] + K[2] + M[2];
    assign rotated[2] = {step_sum[2][14:0], step_sum[2][31:15]}; // <<< 17
    assign sa[3] = sd[2]; assign sb[3] = sb[2] + rotated[2]; assign sc[3] = sb[2]; assign sd[3] = sc[2];

    // Step 3
    assign f_val[3] = (sb[3] & sc[3]) | (~sb[3] & sd[3]);
    assign step_sum[3] = sa[3] + f_val[3] + K[3] + M[3];
    assign rotated[3] = {step_sum[3][9:0], step_sum[3][31:10]}; // <<< 22
    assign sa[4] = sd[3]; assign sb[4] = sb[3] + rotated[3]; assign sc[4] = sb[3]; assign sd[4] = sc[3];

    // Step 4
    assign f_val[4] = (sb[4] & sc[4]) | (~sb[4] & sd[4]);
    assign step_sum[4] = sa[4] + f_val[4] + K[4] + M[4];
    assign rotated[4] = {step_sum[4][24:0], step_sum[4][31:25]}; // <<< 7
    assign sa[5] = sd[4]; assign sb[5] = sb[4] + rotated[4]; assign sc[5] = sb[4]; assign sd[5] = sc[4];

    // Step 5
    assign f_val[5] = (sb[5] & sc[5]) | (~sb[5] & sd[5]);
    assign step_sum[5] = sa[5] + f_val[5] + K[5] + M[5];
    assign rotated[5] = {step_sum[5][19:0], step_sum[5][31:20]}; // <<< 12
    assign sa[6] = sd[5]; assign sb[6] = sb[5] + rotated[5]; assign sc[6] = sb[5]; assign sd[6] = sc[5];

    // Step 6
    assign f_val[6] = (sb[6] & sc[6]) | (~sb[6] & sd[6]);
    assign step_sum[6] = sa[6] + f_val[6] + K[6] + M[6];
    assign rotated[6] = {step_sum[6][14:0], step_sum[6][31:15]}; // <<< 17
    assign sa[7] = sd[6]; assign sb[7] = sb[6] + rotated[6]; assign sc[7] = sb[6]; assign sd[7] = sc[6];

    // Step 7
    assign f_val[7] = (sb[7] & sc[7]) | (~sb[7] & sd[7]);
    assign step_sum[7] = sa[7] + f_val[7] + K[7] + M[7];
    assign rotated[7] = {step_sum[7][9:0], step_sum[7][31:10]}; // <<< 22
    assign sa[8] = sd[7]; assign sb[8] = sb[7] + rotated[7]; assign sc[8] = sb[7]; assign sd[8] = sc[7];

    // Step 8
    assign f_val[8] = (sb[8] & sc[8]) | (~sb[8] & sd[8]);
    assign step_sum[8] = sa[8] + f_val[8] + K[8] + M[8];
    assign rotated[8] = {step_sum[8][24:0], step_sum[8][31:25]}; // <<< 7
    assign sa[9] = sd[8]; assign sb[9] = sb[8] + rotated[8]; assign sc[9] = sb[8]; assign sd[9] = sc[8];

    // Step 9
    assign f_val[9] = (sb[9] & sc[9]) | (~sb[9] & sd[9]);
    assign step_sum[9] = sa[9] + f_val[9] + K[9] + M[9];
    assign rotated[9] = {step_sum[9][19:0], step_sum[9][31:20]}; // <<< 12
    assign sa[10] = sd[9]; assign sb[10] = sb[9] + rotated[9]; assign sc[10] = sb[9]; assign sd[10] = sc[9];

    // Step 10
    assign f_val[10] = (sb[10] & sc[10]) | (~sb[10] & sd[10]);
    assign step_sum[10] = sa[10] + f_val[10] + K[10] + M[10];
    assign rotated[10] = {step_sum[10][14:0], step_sum[10][31:15]}; // <<< 17
    assign sa[11] = sd[10]; assign sb[11] = sb[10] + rotated[10]; assign sc[11] = sb[10]; assign sd[11] = sc[10];

    // Step 11
    assign f_val[11] = (sb[11] & sc[11]) | (~sb[11] & sd[11]);
    assign step_sum[11] = sa[11] + f_val[11] + K[11] + M[11];
    assign rotated[11] = {step_sum[11][9:0], step_sum[11][31:10]}; // <<< 22
    assign sa[12] = sd[11]; assign sb[12] = sb[11] + rotated[11]; assign sc[12] = sb[11]; assign sd[12] = sc[11];

    // Step 12
    assign f_val[12] = (sb[12] & sc[12]) | (~sb[12] & sd[12]);
    assign step_sum[12] = sa[12] + f_val[12] + K[12] + M[12];
    assign rotated[12] = {step_sum[12][24:0], step_sum[12][31:25]}; // <<< 7
    assign sa[13] = sd[12]; assign sb[13] = sb[12] + rotated[12]; assign sc[13] = sb[12]; assign sd[13] = sc[12];

    // Step 13
    assign f_val[13] = (sb[13] & sc[13]) | (~sb[13] & sd[13]);
    assign step_sum[13] = sa[13] + f_val[13] + K[13] + M[13];
    assign rotated[13] = {step_sum[13][19:0], step_sum[13][31:20]}; // <<< 12
    assign sa[14] = sd[13]; assign sb[14] = sb[13] + rotated[13]; assign sc[14] = sb[13]; assign sd[14] = sc[13];

    // Step 14
    assign f_val[14] = (sb[14] & sc[14]) | (~sb[14] & sd[14]);
    assign step_sum[14] = sa[14] + f_val[14] + K[14] + M[14];
    assign rotated[14] = {step_sum[14][14:0], step_sum[14][31:15]}; // <<< 17
    assign sa[15] = sd[14]; assign sb[15] = sb[14] + rotated[14]; assign sc[15] = sb[14]; assign sd[15] = sc[14];

    // Step 15
    assign f_val[15] = (sb[15] & sc[15]) | (~sb[15] & sd[15]);
    assign step_sum[15] = sa[15] + f_val[15] + K[15] + M[15];
    assign rotated[15] = {step_sum[15][9:0], step_sum[15][31:10]}; // <<< 22
    assign sa[16] = sd[15]; assign sb[16] = sb[15] + rotated[15]; assign sc[16] = sb[15]; assign sd[16] = sc[15];

    // Round 2 (steps 16-31): G(B,C,D) = (B & D) | (C & ~D), g = (5*i+1) mod 16
    // Shift amounts: 5, 9, 14, 20, ...

    // Step 16, g=1
    assign f_val[16] = (sb[16] & sd[16]) | (sc[16] & ~sd[16]);
    assign step_sum[16] = sa[16] + f_val[16] + K[16] + M[1];
    assign rotated[16] = {step_sum[16][26:0], step_sum[16][31:27]}; // <<< 5
    assign sa[17] = sd[16]; assign sb[17] = sb[16] + rotated[16]; assign sc[17] = sb[16]; assign sd[17] = sc[16];

    // Step 17, g=6
    assign f_val[17] = (sb[17] & sd[17]) | (sc[17] & ~sd[17]);
    assign step_sum[17] = sa[17] + f_val[17] + K[17] + M[6];
    assign rotated[17] = {step_sum[17][22:0], step_sum[17][31:23]}; // <<< 9
    assign sa[18] = sd[17]; assign sb[18] = sb[17] + rotated[17]; assign sc[18] = sb[17]; assign sd[18] = sc[17];

    // Step 18, g=11
    assign f_val[18] = (sb[18] & sd[18]) | (sc[18] & ~sd[18]);
    assign step_sum[18] = sa[18] + f_val[18] + K[18] + M[11];
    assign rotated[18] = {step_sum[18][17:0], step_sum[18][31:18]}; // <<< 14
    assign sa[19] = sd[18]; assign sb[19] = sb[18] + rotated[18]; assign sc[19] = sb[18]; assign sd[19] = sc[18];

    // Step 19, g=0
    assign f_val[19] = (sb[19] & sd[19]) | (sc[19] & ~sd[19]);
    assign step_sum[19] = sa[19] + f_val[19] + K[19] + M[0];
    assign rotated[19] = {step_sum[19][11:0], step_sum[19][31:12]}; // <<< 20
    assign sa[20] = sd[19]; assign sb[20] = sb[19] + rotated[19]; assign sc[20] = sb[19]; assign sd[20] = sc[19];

    // Step 20, g=5
    assign f_val[20] = (sb[20] & sd[20]) | (sc[20] & ~sd[20]);
    assign step_sum[20] = sa[20] + f_val[20] + K[20] + M[5];
    assign rotated[20] = {step_sum[20][26:0], step_sum[20][31:27]}; // <<< 5
    assign sa[21] = sd[20]; assign sb[21] = sb[20] + rotated[20]; assign sc[21] = sb[20]; assign sd[21] = sc[20];

    // Step 21, g=10
    assign f_val[21] = (sb[21] & sd[21]) | (sc[21] & ~sd[21]);
    assign step_sum[21] = sa[21] + f_val[21] + K[21] + M[10];
    assign rotated[21] = {step_sum[21][22:0], step_sum[21][31:23]}; // <<< 9
    assign sa[22] = sd[21]; assign sb[22] = sb[21] + rotated[21]; assign sc[22] = sb[21]; assign sd[22] = sc[21];

    // Step 22, g=15
    assign f_val[22] = (sb[22] & sd[22]) | (sc[22] & ~sd[22]);
    assign step_sum[22] = sa[22] + f_val[22] + K[22] + M[15];
    assign rotated[22] = {step_sum[22][17:0], step_sum[22][31:18]}; // <<< 14
    assign sa[23] = sd[22]; assign sb[23] = sb[22] + rotated[22]; assign sc[23] = sb[22]; assign sd[23] = sc[22];

    // Step 23, g=4
    assign f_val[23] = (sb[23] & sd[23]) | (sc[23] & ~sd[23]);
    assign step_sum[23] = sa[23] + f_val[23] + K[23] + M[4];
    assign rotated[23] = {step_sum[23][11:0], step_sum[23][31:12]}; // <<< 20
    assign sa[24] = sd[23]; assign sb[24] = sb[23] + rotated[23]; assign sc[24] = sb[23]; assign sd[24] = sc[23];

    // Step 24, g=9
    assign f_val[24] = (sb[24] & sd[24]) | (sc[24] & ~sd[24]);
    assign step_sum[24] = sa[24] + f_val[24] + K[24] + M[9];
    assign rotated[24] = {step_sum[24][26:0], step_sum[24][31:27]}; // <<< 5
    assign sa[25] = sd[24]; assign sb[25] = sb[24] + rotated[24]; assign sc[25] = sb[24]; assign sd[25] = sc[24];

    // Step 25, g=14
    assign f_val[25] = (sb[25] & sd[25]) | (sc[25] & ~sd[25]);
    assign step_sum[25] = sa[25] + f_val[25] + K[25] + M[14];
    assign rotated[25] = {step_sum[25][22:0], step_sum[25][31:23]}; // <<< 9
    assign sa[26] = sd[25]; assign sb[26] = sb[25] + rotated[25]; assign sc[26] = sb[25]; assign sd[26] = sc[25];

    // Step 26, g=3
    assign f_val[26] = (sb[26] & sd[26]) | (sc[26] & ~sd[26]);
    assign step_sum[26] = sa[26] + f_val[26] + K[26] + M[3];
    assign rotated[26] = {step_sum[26][17:0], step_sum[26][31:18]}; // <<< 14
    assign sa[27] = sd[26]; assign sb[27] = sb[26] + rotated[26]; assign sc[27] = sb[26]; assign sd[27] = sc[26];

    // Step 27, g=8
    assign f_val[27] = (sb[27] & sd[27]) | (sc[27] & ~sd[27]);
    assign step_sum[27] = sa[27] + f_val[27] + K[27] + M[8];
    assign rotated[27] = {step_sum[27][11:0], step_sum[27][31:12]}; // <<< 20
    assign sa[28] = sd[27]; assign sb[28] = sb[27] + rotated[27]; assign sc[28] = sb[27]; assign sd[28] = sc[27];

    // Step 28, g=13
    assign f_val[28] = (sb[28] & sd[28]) | (sc[28] & ~sd[28]);
    assign step_sum[28] = sa[28] + f_val[28] + K[28] + M[13];
    assign rotated[28] = {step_sum[28][26:0], step_sum[28][31:27]}; // <<< 5
    assign sa[29] = sd[28]; assign sb[29] = sb[28] + rotated[28]; assign sc[29] = sb[28]; assign sd[29] = sc[28];

    // Step 29, g=2
    assign f_val[29] = (sb[29] & sd[29]) | (sc[29] & ~sd[29]);
    assign step_sum[29] = sa[29] + f_val[29] + K[29] + M[2];
    assign rotated[29] = {step_sum[29][22:0], step_sum[29][31:23]}; // <<< 9
    assign sa[30] = sd[29]; assign sb[30] = sb[29] + rotated[29]; assign sc[30] = sb[29]; assign sd[30] = sc[29];

    // Step 30, g=7
    assign f_val[30] = (sb[30] & sd[30]) | (sc[30] & ~sd[30]);
    assign step_sum[30] = sa[30] + f_val[30] + K[30] + M[7];
    assign rotated[30] = {step_sum[30][17:0], step_sum[30][31:18]}; // <<< 14
    assign sa[31] = sd[30]; assign sb[31] = sb[30] + rotated[30]; assign sc[31] = sb[30]; assign sd[31] = sc[30];

    // Step 31, g=12
    assign f_val[31] = (sb[31] & sd[31]) | (sc[31] & ~sd[31]);
    assign step_sum[31] = sa[31] + f_val[31] + K[31] + M[12];
    assign rotated[31] = {step_sum[31][11:0], step_sum[31][31:12]}; // <<< 20
    assign sa[32] = sd[31]; assign sb[32] = sb[31] + rotated[31]; assign sc[32] = sb[31]; assign sd[32] = sc[31];

    // Round 3 (steps 32-47): H(B,C,D) = B ^ C ^ D, g = (3*i+5) mod 16
    // Shift amounts: 4, 11, 16, 23, ...

    // Step 32, g=5
    assign f_val[32] = sb[32] ^ sc[32] ^ sd[32];
    assign step_sum[32] = sa[32] + f_val[32] + K[32] + M[5];
    assign rotated[32] = {step_sum[32][27:0], step_sum[32][31:28]}; // <<< 4
    assign sa[33] = sd[32]; assign sb[33] = sb[32] + rotated[32]; assign sc[33] = sb[32]; assign sd[33] = sc[32];

    // Step 33, g=8
    assign f_val[33] = sb[33] ^ sc[33] ^ sd[33];
    assign step_sum[33] = sa[33] + f_val[33] + K[33] + M[8];
    assign rotated[33] = {step_sum[33][20:0], step_sum[33][31:21]}; // <<< 11
    assign sa[34] = sd[33]; assign sb[34] = sb[33] + rotated[33]; assign sc[34] = sb[33]; assign sd[34] = sc[33];

    // Step 34, g=11
    assign f_val[34] = sb[34] ^ sc[34] ^ sd[34];
    assign step_sum[34] = sa[34] + f_val[34] + K[34] + M[11];
    assign rotated[34] = {step_sum[34][15:0], step_sum[34][31:16]}; // <<< 16
    assign sa[35] = sd[34]; assign sb[35] = sb[34] + rotated[34]; assign sc[35] = sb[34]; assign sd[35] = sc[34];

    // Step 35, g=14
    assign f_val[35] = sb[35] ^ sc[35] ^ sd[35];
    assign step_sum[35] = sa[35] + f_val[35] + K[35] + M[14];
    assign rotated[35] = {step_sum[35][8:0], step_sum[35][31:9]}; // <<< 23
    assign sa[36] = sd[35]; assign sb[36] = sb[35] + rotated[35]; assign sc[36] = sb[35]; assign sd[36] = sc[35];

    // Step 36, g=1
    assign f_val[36] = sb[36] ^ sc[36] ^ sd[36];
    assign step_sum[36] = sa[36] + f_val[36] + K[36] + M[1];
    assign rotated[36] = {step_sum[36][27:0], step_sum[36][31:28]}; // <<< 4
    assign sa[37] = sd[36]; assign sb[37] = sb[36] + rotated[36]; assign sc[37] = sb[36]; assign sd[37] = sc[36];

    // Step 37, g=4
    assign f_val[37] = sb[37] ^ sc[37] ^ sd[37];
    assign step_sum[37] = sa[37] + f_val[37] + K[37] + M[4];
    assign rotated[37] = {step_sum[37][20:0], step_sum[37][31:21]}; // <<< 11
    assign sa[38] = sd[37]; assign sb[38] = sb[37] + rotated[37]; assign sc[38] = sb[37]; assign sd[38] = sc[37];

    // Step 38, g=7
    assign f_val[38] = sb[38] ^ sc[38] ^ sd[38];
    assign step_sum[38] = sa[38] + f_val[38] + K[38] + M[7];
    assign rotated[38] = {step_sum[38][15:0], step_sum[38][31:16]}; // <<< 16
    assign sa[39] = sd[38]; assign sb[39] = sb[38] + rotated[38]; assign sc[39] = sb[38]; assign sd[39] = sc[38];

    // Step 39, g=10
    assign f_val[39] = sb[39] ^ sc[39] ^ sd[39];
    assign step_sum[39] = sa[39] + f_val[39] + K[39] + M[10];
    assign rotated[39] = {step_sum[39][8:0], step_sum[39][31:9]}; // <<< 23
    assign sa[40] = sd[39]; assign sb[40] = sb[39] + rotated[39]; assign sc[40] = sb[39]; assign sd[40] = sc[39];

    // Step 40, g=13
    assign f_val[40] = sb[40] ^ sc[40] ^ sd[40];
    assign step_sum[40] = sa[40] + f_val[40] + K[40] + M[13];
    assign rotated[40] = {step_sum[40][27:0], step_sum[40][31:28]}; // <<< 4
    assign sa[41] = sd[40]; assign sb[41] = sb[40] + rotated[40]; assign sc[41] = sb[40]; assign sd[41] = sc[40];

    // Step 41, g=0
    assign f_val[41] = sb[41] ^ sc[41] ^ sd[41];
    assign step_sum[41] = sa[41] + f_val[41] + K[41] + M[0];
    assign rotated[41] = {step_sum[41][20:0], step_sum[41][31:21]}; // <<< 11
    assign sa[42] = sd[41]; assign sb[42] = sb[41] + rotated[41]; assign sc[42] = sb[41]; assign sd[42] = sc[41];

    // Step 42, g=3
    assign f_val[42] = sb[42] ^ sc[42] ^ sd[42];
    assign step_sum[42] = sa[42] + f_val[42] + K[42] + M[3];
    assign rotated[42] = {step_sum[42][15:0], step_sum[42][31:16]}; // <<< 16
    assign sa[43] = sd[42]; assign sb[43] = sb[42] + rotated[42]; assign sc[43] = sb[42]; assign sd[43] = sc[42];

    // Step 43, g=6
    assign f_val[43] = sb[43] ^ sc[43] ^ sd[43];
    assign step_sum[43] = sa[43] + f_val[43] + K[43] + M[6];
    assign rotated[43] = {step_sum[43][8:0], step_sum[43][31:9]}; // <<< 23
    assign sa[44] = sd[43]; assign sb[44] = sb[43] + rotated[43]; assign sc[44] = sb[43]; assign sd[44] = sc[43];

    // Step 44, g=9
    assign f_val[44] = sb[44] ^ sc[44] ^ sd[44];
    assign step_sum[44] = sa[44] + f_val[44] + K[44] + M[9];
    assign rotated[44] = {step_sum[44][27:0], step_sum[44][31:28]}; // <<< 4
    assign sa[45] = sd[44]; assign sb[45] = sb[44] + rotated[44]; assign sc[45] = sb[44]; assign sd[45] = sc[44];

    // Step 45, g=12
    assign f_val[45] = sb[45] ^ sc[45] ^ sd[45];
    assign step_sum[45] = sa[45] + f_val[45] + K[45] + M[12];
    assign rotated[45] = {step_sum[45][20:0], step_sum[45][31:21]}; // <<< 11
    assign sa[46] = sd[45]; assign sb[46] = sb[45] + rotated[45]; assign sc[46] = sb[45]; assign sd[46] = sc[45];

    // Step 46, g=15
    assign f_val[46] = sb[46] ^ sc[46] ^ sd[46];
    assign step_sum[46] = sa[46] + f_val[46] + K[46] + M[15];
    assign rotated[46] = {step_sum[46][15:0], step_sum[46][31:16]}; // <<< 16
    assign sa[47] = sd[46]; assign sb[47] = sb[46] + rotated[46]; assign sc[47] = sb[46]; assign sd[47] = sc[46];

    // Step 47, g=2
    assign f_val[47] = sb[47] ^ sc[47] ^ sd[47];
    assign step_sum[47] = sa[47] + f_val[47] + K[47] + M[2];
    assign rotated[47] = {step_sum[47][8:0], step_sum[47][31:9]}; // <<< 23
    assign sa[48] = sd[47]; assign sb[48] = sb[47] + rotated[47]; assign sc[48] = sb[47]; assign sd[48] = sc[47];

    // Round 4 (steps 48-63): I(B,C,D) = C ^ (B | ~D), g = (7*i) mod 16
    // Shift amounts: 6, 10, 15, 21, ...

    // Step 48, g=0
    assign f_val[48] = sc[48] ^ (sb[48] | ~sd[48]);
    assign step_sum[48] = sa[48] + f_val[48] + K[48] + M[0];
    assign rotated[48] = {step_sum[48][25:0], step_sum[48][31:26]}; // <<< 6
    assign sa[49] = sd[48]; assign sb[49] = sb[48] + rotated[48]; assign sc[49] = sb[48]; assign sd[49] = sc[48];

    // Step 49, g=7
    assign f_val[49] = sc[49] ^ (sb[49] | ~sd[49]);
    assign step_sum[49] = sa[49] + f_val[49] + K[49] + M[7];
    assign rotated[49] = {step_sum[49][21:0], step_sum[49][31:22]}; // <<< 10
    assign sa[50] = sd[49]; assign sb[50] = sb[49] + rotated[49]; assign sc[50] = sb[49]; assign sd[50] = sc[49];

    // Step 50, g=14
    assign f_val[50] = sc[50] ^ (sb[50] | ~sd[50]);
    assign step_sum[50] = sa[50] + f_val[50] + K[50] + M[14];
    assign rotated[50] = {step_sum[50][16:0], step_sum[50][31:17]}; // <<< 15
    assign sa[51] = sd[50]; assign sb[51] = sb[50] + rotated[50]; assign sc[51] = sb[50]; assign sd[51] = sc[50];

    // Step 51, g=5
    assign f_val[51] = sc[51] ^ (sb[51] | ~sd[51]);
    assign step_sum[51] = sa[51] + f_val[51] + K[51] + M[5];
    assign rotated[51] = {step_sum[51][10:0], step_sum[51][31:11]}; // <<< 21
    assign sa[52] = sd[51]; assign sb[52] = sb[51] + rotated[51]; assign sc[52] = sb[51]; assign sd[52] = sc[51];

    // Step 52, g=12
    assign f_val[52] = sc[52] ^ (sb[52] | ~sd[52]);
    assign step_sum[52] = sa[52] + f_val[52] + K[52] + M[12];
    assign rotated[52] = {step_sum[52][25:0], step_sum[52][31:26]}; // <<< 6
    assign sa[53] = sd[52]; assign sb[53] = sb[52] + rotated[52]; assign sc[53] = sb[52]; assign sd[53] = sc[52];

    // Step 53, g=3
    assign f_val[53] = sc[53] ^ (sb[53] | ~sd[53]);
    assign step_sum[53] = sa[53] + f_val[53] + K[53] + M[3];
    assign rotated[53] = {step_sum[53][21:0], step_sum[53][31:22]}; // <<< 10
    assign sa[54] = sd[53]; assign sb[54] = sb[53] + rotated[53]; assign sc[54] = sb[53]; assign sd[54] = sc[53];

    // Step 54, g=10
    assign f_val[54] = sc[54] ^ (sb[54] | ~sd[54]);
    assign step_sum[54] = sa[54] + f_val[54] + K[54] + M[10];
    assign rotated[54] = {step_sum[54][16:0], step_sum[54][31:17]}; // <<< 15
    assign sa[55] = sd[54]; assign sb[55] = sb[54] + rotated[54]; assign sc[55] = sb[54]; assign sd[55] = sc[54];

    // Step 55, g=1
    assign f_val[55] = sc[55] ^ (sb[55] | ~sd[55]);
    assign step_sum[55] = sa[55] + f_val[55] + K[55] + M[1];
    assign rotated[55] = {step_sum[55][10:0], step_sum[55][31:11]}; // <<< 21
    assign sa[56] = sd[55]; assign sb[56] = sb[55] + rotated[55]; assign sc[56] = sb[55]; assign sd[56] = sc[55];

    // Step 56, g=8
    assign f_val[56] = sc[56] ^ (sb[56] | ~sd[56]);
    assign step_sum[56] = sa[56] + f_val[56] + K[56] + M[8];
    assign rotated[56] = {step_sum[56][25:0], step_sum[56][31:26]}; // <<< 6
    assign sa[57] = sd[56]; assign sb[57] = sb[56] + rotated[56]; assign sc[57] = sb[56]; assign sd[57] = sc[56];

    // Step 57, g=15
    assign f_val[57] = sc[57] ^ (sb[57] | ~sd[57]);
    assign step_sum[57] = sa[57] + f_val[57] + K[57] + M[15];
    assign rotated[57] = {step_sum[57][21:0], step_sum[57][31:22]}; // <<< 10
    assign sa[58] = sd[57]; assign sb[58] = sb[57] + rotated[57]; assign sc[58] = sb[57]; assign sd[58] = sc[57];

    // Step 58, g=6
    assign f_val[58] = sc[58] ^ (sb[58] | ~sd[58]);
    assign step_sum[58] = sa[58] + f_val[58] + K[58] + M[6];
    assign rotated[58] = {step_sum[58][16:0], step_sum[58][31:17]}; // <<< 15
    assign sa[59] = sd[58]; assign sb[59] = sb[58] + rotated[58]; assign sc[59] = sb[58]; assign sd[59] = sc[58];

    // Step 59, g=13
    assign f_val[59] = sc[59] ^ (sb[59] | ~sd[59]);
    assign step_sum[59] = sa[59] + f_val[59] + K[59] + M[13];
    assign rotated[59] = {step_sum[59][10:0], step_sum[59][31:11]}; // <<< 21
    assign sa[60] = sd[59]; assign sb[60] = sb[59] + rotated[59]; assign sc[60] = sb[59]; assign sd[60] = sc[59];

    // Step 60, g=4
    assign f_val[60] = sc[60] ^ (sb[60] | ~sd[60]);
    assign step_sum[60] = sa[60] + f_val[60] + K[60] + M[4];
    assign rotated[60] = {step_sum[60][25:0], step_sum[60][31:26]}; // <<< 6
    assign sa[61] = sd[60]; assign sb[61] = sb[60] + rotated[60]; assign sc[61] = sb[60]; assign sd[61] = sc[60];

    // Step 61, g=11
    assign f_val[61] = sc[61] ^ (sb[61] | ~sd[61]);
    assign step_sum[61] = sa[61] + f_val[61] + K[61] + M[11];
    assign rotated[61] = {step_sum[61][21:0], step_sum[61][31:22]}; // <<< 10
    assign sa[62] = sd[61]; assign sb[62] = sb[61] + rotated[61]; assign sc[62] = sb[61]; assign sd[62] = sc[61];

    // Step 62, g=2
    assign f_val[62] = sc[62] ^ (sb[62] | ~sd[62]);
    assign step_sum[62] = sa[62] + f_val[62] + K[62] + M[2];
    assign rotated[62] = {step_sum[62][16:0], step_sum[62][31:17]}; // <<< 15
    assign sa[63] = sd[62]; assign sb[63] = sb[62] + rotated[62]; assign sc[63] = sb[62]; assign sd[63] = sc[62];

    // Step 63, g=9
    assign f_val[63] = sc[63] ^ (sb[63] | ~sd[63]);
    assign step_sum[63] = sa[63] + f_val[63] + K[63] + M[9];
    assign rotated[63] = {step_sum[63][10:0], step_sum[63][31:11]}; // <<< 21
    assign sa[64] = sd[63]; assign sb[64] = sb[63] + rotated[63]; assign sc[64] = sb[63]; assign sd[64] = sc[63];

    // Final addition (add initial values)
    assign a_out = a0 + sa[64];
    assign b_out = b0 + sb[64];
    assign c_out = c0 + sc[64];
    assign d_out = d0 + sd[64];

endmodule
