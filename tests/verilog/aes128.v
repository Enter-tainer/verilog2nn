// T14: Full AES-128 encryption - combinational logic, all 10 rounds unrolled
// Input: 128-bit plaintext + 128-bit key
// Output: 128-bit ciphertext
// Total input: 256 bits, output: 128 bits

module aes128(
    input [127:0] plaintext,
    input [127:0] key,
    output [127:0] ciphertext
);

    // AES S-Box as a combinational function
    function [7:0] sbox;
        input [7:0] in;
        case (in)
            8'h00: sbox = 8'h63; 8'h01: sbox = 8'h7c; 8'h02: sbox = 8'h77; 8'h03: sbox = 8'h7b;
            8'h04: sbox = 8'hf2; 8'h05: sbox = 8'h6b; 8'h06: sbox = 8'h6f; 8'h07: sbox = 8'hc5;
            8'h08: sbox = 8'h30; 8'h09: sbox = 8'h01; 8'h0a: sbox = 8'h67; 8'h0b: sbox = 8'h2b;
            8'h0c: sbox = 8'hfe; 8'h0d: sbox = 8'hd7; 8'h0e: sbox = 8'hab; 8'h0f: sbox = 8'h76;
            8'h10: sbox = 8'hca; 8'h11: sbox = 8'h82; 8'h12: sbox = 8'hc9; 8'h13: sbox = 8'h7d;
            8'h14: sbox = 8'hfa; 8'h15: sbox = 8'h59; 8'h16: sbox = 8'h47; 8'h17: sbox = 8'hf0;
            8'h18: sbox = 8'had; 8'h19: sbox = 8'hd4; 8'h1a: sbox = 8'ha2; 8'h1b: sbox = 8'haf;
            8'h1c: sbox = 8'h9c; 8'h1d: sbox = 8'ha4; 8'h1e: sbox = 8'h72; 8'h1f: sbox = 8'hc0;
            8'h20: sbox = 8'hb7; 8'h21: sbox = 8'hfd; 8'h22: sbox = 8'h93; 8'h23: sbox = 8'h26;
            8'h24: sbox = 8'h36; 8'h25: sbox = 8'h3f; 8'h26: sbox = 8'hf7; 8'h27: sbox = 8'hcc;
            8'h28: sbox = 8'h34; 8'h29: sbox = 8'ha5; 8'h2a: sbox = 8'he5; 8'h2b: sbox = 8'hf1;
            8'h2c: sbox = 8'h71; 8'h2d: sbox = 8'hd8; 8'h2e: sbox = 8'h31; 8'h2f: sbox = 8'h15;
            8'h30: sbox = 8'h04; 8'h31: sbox = 8'hc7; 8'h32: sbox = 8'h23; 8'h33: sbox = 8'hc3;
            8'h34: sbox = 8'h18; 8'h35: sbox = 8'h96; 8'h36: sbox = 8'h05; 8'h37: sbox = 8'h9a;
            8'h38: sbox = 8'h07; 8'h39: sbox = 8'h12; 8'h3a: sbox = 8'h80; 8'h3b: sbox = 8'he2;
            8'h3c: sbox = 8'heb; 8'h3d: sbox = 8'h27; 8'h3e: sbox = 8'hb2; 8'h3f: sbox = 8'h75;
            8'h40: sbox = 8'h09; 8'h41: sbox = 8'h83; 8'h42: sbox = 8'h2c; 8'h43: sbox = 8'h1a;
            8'h44: sbox = 8'h1b; 8'h45: sbox = 8'h6e; 8'h46: sbox = 8'h5a; 8'h47: sbox = 8'ha0;
            8'h48: sbox = 8'h52; 8'h49: sbox = 8'h3b; 8'h4a: sbox = 8'hd6; 8'h4b: sbox = 8'hb3;
            8'h4c: sbox = 8'h29; 8'h4d: sbox = 8'he3; 8'h4e: sbox = 8'h2f; 8'h4f: sbox = 8'h84;
            8'h50: sbox = 8'h53; 8'h51: sbox = 8'hd1; 8'h52: sbox = 8'h00; 8'h53: sbox = 8'hed;
            8'h54: sbox = 8'h20; 8'h55: sbox = 8'hfc; 8'h56: sbox = 8'hb1; 8'h57: sbox = 8'h5b;
            8'h58: sbox = 8'h6a; 8'h59: sbox = 8'hcb; 8'h5a: sbox = 8'hbe; 8'h5b: sbox = 8'h39;
            8'h5c: sbox = 8'h4a; 8'h5d: sbox = 8'h4c; 8'h5e: sbox = 8'h58; 8'h5f: sbox = 8'hcf;
            8'h60: sbox = 8'hd0; 8'h61: sbox = 8'hef; 8'h62: sbox = 8'haa; 8'h63: sbox = 8'hfb;
            8'h64: sbox = 8'h43; 8'h65: sbox = 8'h4d; 8'h66: sbox = 8'h33; 8'h67: sbox = 8'h85;
            8'h68: sbox = 8'h45; 8'h69: sbox = 8'hf9; 8'h6a: sbox = 8'h02; 8'h6b: sbox = 8'h7f;
            8'h6c: sbox = 8'h50; 8'h6d: sbox = 8'h3c; 8'h6e: sbox = 8'h9f; 8'h6f: sbox = 8'ha8;
            8'h70: sbox = 8'h51; 8'h71: sbox = 8'ha3; 8'h72: sbox = 8'h40; 8'h73: sbox = 8'h8f;
            8'h74: sbox = 8'h92; 8'h75: sbox = 8'h9d; 8'h76: sbox = 8'h38; 8'h77: sbox = 8'hf5;
            8'h78: sbox = 8'hbc; 8'h79: sbox = 8'hb6; 8'h7a: sbox = 8'hda; 8'h7b: sbox = 8'h21;
            8'h7c: sbox = 8'h10; 8'h7d: sbox = 8'hff; 8'h7e: sbox = 8'hf3; 8'h7f: sbox = 8'hd2;
            8'h80: sbox = 8'hcd; 8'h81: sbox = 8'h0c; 8'h82: sbox = 8'h13; 8'h83: sbox = 8'hec;
            8'h84: sbox = 8'h5f; 8'h85: sbox = 8'h97; 8'h86: sbox = 8'h44; 8'h87: sbox = 8'h17;
            8'h88: sbox = 8'hc4; 8'h89: sbox = 8'ha7; 8'h8a: sbox = 8'h7e; 8'h8b: sbox = 8'h3d;
            8'h8c: sbox = 8'h64; 8'h8d: sbox = 8'h5d; 8'h8e: sbox = 8'h19; 8'h8f: sbox = 8'h73;
            8'h90: sbox = 8'h60; 8'h91: sbox = 8'h81; 8'h92: sbox = 8'h4f; 8'h93: sbox = 8'hdc;
            8'h94: sbox = 8'h22; 8'h95: sbox = 8'h2a; 8'h96: sbox = 8'h90; 8'h97: sbox = 8'h88;
            8'h98: sbox = 8'h46; 8'h99: sbox = 8'hee; 8'h9a: sbox = 8'hb8; 8'h9b: sbox = 8'h14;
            8'h9c: sbox = 8'hde; 8'h9d: sbox = 8'h5e; 8'h9e: sbox = 8'h0b; 8'h9f: sbox = 8'hdb;
            8'ha0: sbox = 8'he0; 8'ha1: sbox = 8'h32; 8'ha2: sbox = 8'h3a; 8'ha3: sbox = 8'h0a;
            8'ha4: sbox = 8'h49; 8'ha5: sbox = 8'h06; 8'ha6: sbox = 8'h24; 8'ha7: sbox = 8'h5c;
            8'ha8: sbox = 8'hc2; 8'ha9: sbox = 8'hd3; 8'haa: sbox = 8'hac; 8'hab: sbox = 8'h62;
            8'hac: sbox = 8'h91; 8'had: sbox = 8'h95; 8'hae: sbox = 8'he4; 8'haf: sbox = 8'h79;
            8'hb0: sbox = 8'he7; 8'hb1: sbox = 8'hc8; 8'hb2: sbox = 8'h37; 8'hb3: sbox = 8'h6d;
            8'hb4: sbox = 8'h8d; 8'hb5: sbox = 8'hd5; 8'hb6: sbox = 8'h4e; 8'hb7: sbox = 8'ha9;
            8'hb8: sbox = 8'h6c; 8'hb9: sbox = 8'h56; 8'hba: sbox = 8'hf4; 8'hbb: sbox = 8'hea;
            8'hbc: sbox = 8'h65; 8'hbd: sbox = 8'h7a; 8'hbe: sbox = 8'hae; 8'hbf: sbox = 8'h08;
            8'hc0: sbox = 8'hba; 8'hc1: sbox = 8'h78; 8'hc2: sbox = 8'h25; 8'hc3: sbox = 8'h2e;
            8'hc4: sbox = 8'h1c; 8'hc5: sbox = 8'ha6; 8'hc6: sbox = 8'hb4; 8'hc7: sbox = 8'hc6;
            8'hc8: sbox = 8'he8; 8'hc9: sbox = 8'hdd; 8'hca: sbox = 8'h74; 8'hcb: sbox = 8'h1f;
            8'hcc: sbox = 8'h4b; 8'hcd: sbox = 8'hbd; 8'hce: sbox = 8'h8b; 8'hcf: sbox = 8'h8a;
            8'hd0: sbox = 8'h70; 8'hd1: sbox = 8'h3e; 8'hd2: sbox = 8'hb5; 8'hd3: sbox = 8'h66;
            8'hd4: sbox = 8'h48; 8'hd5: sbox = 8'h03; 8'hd6: sbox = 8'hf6; 8'hd7: sbox = 8'h0e;
            8'hd8: sbox = 8'h61; 8'hd9: sbox = 8'h35; 8'hda: sbox = 8'h57; 8'hdb: sbox = 8'hb9;
            8'hdc: sbox = 8'h86; 8'hdd: sbox = 8'hc1; 8'hde: sbox = 8'h1d; 8'hdf: sbox = 8'h9e;
            8'he0: sbox = 8'he1; 8'he1: sbox = 8'hf8; 8'he2: sbox = 8'h98; 8'he3: sbox = 8'h11;
            8'he4: sbox = 8'h69; 8'he5: sbox = 8'hd9; 8'he6: sbox = 8'h8e; 8'he7: sbox = 8'h94;
            8'he8: sbox = 8'h9b; 8'he9: sbox = 8'h1e; 8'hea: sbox = 8'h87; 8'heb: sbox = 8'he9;
            8'hec: sbox = 8'hce; 8'hed: sbox = 8'h55; 8'hee: sbox = 8'h28; 8'hef: sbox = 8'hdf;
            8'hf0: sbox = 8'h8c; 8'hf1: sbox = 8'ha1; 8'hf2: sbox = 8'h89; 8'hf3: sbox = 8'h0d;
            8'hf4: sbox = 8'hbf; 8'hf5: sbox = 8'he6; 8'hf6: sbox = 8'h42; 8'hf7: sbox = 8'h68;
            8'hf8: sbox = 8'h41; 8'hf9: sbox = 8'h99; 8'hfa: sbox = 8'h2d; 8'hfb: sbox = 8'h0f;
            8'hfc: sbox = 8'hb0; 8'hfd: sbox = 8'h54; 8'hfe: sbox = 8'hbb; 8'hff: sbox = 8'h16;
        endcase
    endfunction

    // xtime: multiply by 2 in GF(2^8)
    function [7:0] xtime;
        input [7:0] in;
        xtime = {in[6:0], 1'b0} ^ (8'h1b & {8{in[7]}});
    endfunction

    // Key expansion: generate 11 round keys (key0 = original key)
    // Round constants
    wire [7:0] rcon [1:10];
    assign rcon[1]  = 8'h01; assign rcon[2]  = 8'h02;
    assign rcon[3]  = 8'h04; assign rcon[4]  = 8'h08;
    assign rcon[5]  = 8'h10; assign rcon[6]  = 8'h20;
    assign rcon[7]  = 8'h40; assign rcon[8]  = 8'h80;
    assign rcon[9]  = 8'h1b; assign rcon[10] = 8'h36;

    // Round keys: rk[0] = original key, rk[1..10] = expanded keys
    wire [127:0] rk [0:10];
    assign rk[0] = key;

    // Key expansion - each round key derived from previous
    // Key is stored as 4 columns of 32 bits each (big-endian byte order within state)
    // State bytes: key[127:120]=s00, key[119:112]=s10, key[111:104]=s20, key[103:96]=s30
    //              key[95:88]=s01, ... key[7:0]=s33

    // For key expansion:
    // w[4i] = w[4(i-1)] ^ SubWord(RotWord(w[4i-1])) ^ Rcon[i]
    // w[4i+1] = w[4(i-1)+1] ^ w[4i]
    // w[4i+2] = w[4(i-1)+2] ^ w[4i+1]
    // w[4i+3] = w[4(i-1)+3] ^ w[4i+2]

    genvar ri;
    generate
        for (ri = 1; ri <= 10; ri = ri + 1) begin : key_exp
            // Previous round key words
            wire [31:0] prev_w0 = rk[ri-1][127:96];
            wire [31:0] prev_w1 = rk[ri-1][95:64];
            wire [31:0] prev_w2 = rk[ri-1][63:32];
            wire [31:0] prev_w3 = rk[ri-1][31:0];

            // RotWord(prev_w3): rotate left by 1 byte
            wire [31:0] rot_w3 = {prev_w3[23:16], prev_w3[15:8], prev_w3[7:0], prev_w3[31:24]};

            // SubWord(rot_w3): apply S-Box to each byte
            wire [31:0] sub_rot = {sbox(rot_w3[31:24]), sbox(rot_w3[23:16]),
                                   sbox(rot_w3[15:8]), sbox(rot_w3[7:0])};

            // XOR with Rcon
            wire [31:0] rcon_word = {rcon[ri], 24'h0};

            wire [31:0] new_w0 = prev_w0 ^ sub_rot ^ rcon_word;
            wire [31:0] new_w1 = prev_w1 ^ new_w0;
            wire [31:0] new_w2 = prev_w2 ^ new_w1;
            wire [31:0] new_w3 = prev_w3 ^ new_w2;

            assign rk[ri] = {new_w0, new_w1, new_w2, new_w3};
        end
    endgenerate

    // AES state after each round: state[0] = after initial AddRoundKey
    wire [127:0] state [0:10];

    // Initial AddRoundKey
    assign state[0] = plaintext ^ rk[0];

    // SubBytes: apply S-Box to each byte of state
    // ShiftRows: cyclically shift rows of the state
    // State byte layout (column-major, big-endian):
    //   s00 s01 s02 s03     byte positions: [127:120] [95:88]  [63:56]  [31:24]
    //   s10 s11 s12 s13                     [119:112] [87:80]  [55:48]  [23:16]
    //   s20 s21 s22 s23                     [111:104] [79:72]  [47:40]  [15:8]
    //   s30 s31 s32 s33                     [103:96]  [71:64]  [39:32]  [7:0]
    // ShiftRows:
    //   Row 0: no shift
    //   Row 1: shift left by 1
    //   Row 2: shift left by 2
    //   Row 3: shift left by 3

    // MixColumns: mix each column using GF(2^8) arithmetic
    // For each column [s0, s1, s2, s3]:
    //   r0 = 2*s0 ^ 3*s1 ^ s2 ^ s3
    //   r1 = s0 ^ 2*s1 ^ 3*s2 ^ s3
    //   r2 = s0 ^ s1 ^ 2*s2 ^ 3*s3
    //   r3 = 3*s0 ^ s1 ^ s2 ^ 2*s3

    generate
        for (ri = 1; ri <= 10; ri = ri + 1) begin : aes_rounds
            // Extract state bytes
            wire [7:0] s00 = state[ri-1][127:120];
            wire [7:0] s10 = state[ri-1][119:112];
            wire [7:0] s20 = state[ri-1][111:104];
            wire [7:0] s30 = state[ri-1][103:96];
            wire [7:0] s01 = state[ri-1][95:88];
            wire [7:0] s11 = state[ri-1][87:80];
            wire [7:0] s21 = state[ri-1][79:72];
            wire [7:0] s31 = state[ri-1][71:64];
            wire [7:0] s02 = state[ri-1][63:56];
            wire [7:0] s12 = state[ri-1][55:48];
            wire [7:0] s22 = state[ri-1][47:40];
            wire [7:0] s32 = state[ri-1][39:32];
            wire [7:0] s03 = state[ri-1][31:24];
            wire [7:0] s13 = state[ri-1][23:16];
            wire [7:0] s23 = state[ri-1][15:8];
            wire [7:0] s33 = state[ri-1][7:0];

            // SubBytes
            wire [7:0] sb00 = sbox(s00); wire [7:0] sb10 = sbox(s10);
            wire [7:0] sb20 = sbox(s20); wire [7:0] sb30 = sbox(s30);
            wire [7:0] sb01 = sbox(s01); wire [7:0] sb11 = sbox(s11);
            wire [7:0] sb21 = sbox(s21); wire [7:0] sb31 = sbox(s31);
            wire [7:0] sb02 = sbox(s02); wire [7:0] sb12 = sbox(s12);
            wire [7:0] sb22 = sbox(s22); wire [7:0] sb32 = sbox(s32);
            wire [7:0] sb03 = sbox(s03); wire [7:0] sb13 = sbox(s13);
            wire [7:0] sb23 = sbox(s23); wire [7:0] sb33 = sbox(s33);

            // ShiftRows
            // Row 0 (no shift): sb00, sb01, sb02, sb03
            // Row 1 (shift 1):  sb11, sb12, sb13, sb10
            // Row 2 (shift 2):  sb22, sb23, sb20, sb21
            // Row 3 (shift 3):  sb33, sb30, sb31, sb32
            wire [7:0] sr00 = sb00; wire [7:0] sr10 = sb11;
            wire [7:0] sr20 = sb22; wire [7:0] sr30 = sb33;
            wire [7:0] sr01 = sb01; wire [7:0] sr11 = sb12;
            wire [7:0] sr21 = sb23; wire [7:0] sr31 = sb30;
            wire [7:0] sr02 = sb02; wire [7:0] sr12 = sb13;
            wire [7:0] sr22 = sb20; wire [7:0] sr32 = sb31;
            wire [7:0] sr03 = sb03; wire [7:0] sr13 = sb10;
            wire [7:0] sr23 = sb21; wire [7:0] sr32_3 = sb32;

            // For last round (round 10): skip MixColumns
            if (ri < 10) begin : full_round
                // MixColumns on each column
                // Column 0
                wire [7:0] mc00 = xtime(sr00) ^ (xtime(sr10) ^ sr10) ^ sr20 ^ sr30;
                wire [7:0] mc10 = sr00 ^ xtime(sr10) ^ (xtime(sr20) ^ sr20) ^ sr30;
                wire [7:0] mc20 = sr00 ^ sr10 ^ xtime(sr20) ^ (xtime(sr30) ^ sr30);
                wire [7:0] mc30 = (xtime(sr00) ^ sr00) ^ sr10 ^ sr20 ^ xtime(sr30);

                // Column 1
                wire [7:0] mc01 = xtime(sr01) ^ (xtime(sr11) ^ sr11) ^ sr21 ^ sr31;
                wire [7:0] mc11 = sr01 ^ xtime(sr11) ^ (xtime(sr21) ^ sr21) ^ sr31;
                wire [7:0] mc21 = sr01 ^ sr11 ^ xtime(sr21) ^ (xtime(sr31) ^ sr31);
                wire [7:0] mc31 = (xtime(sr01) ^ sr01) ^ sr11 ^ sr21 ^ xtime(sr31);

                // Column 2
                wire [7:0] mc02 = xtime(sr02) ^ (xtime(sr12) ^ sr12) ^ sr22 ^ sr32;
                wire [7:0] mc12 = sr02 ^ xtime(sr12) ^ (xtime(sr22) ^ sr22) ^ sr32;
                wire [7:0] mc22 = sr02 ^ sr12 ^ xtime(sr22) ^ (xtime(sr32) ^ sr32);
                wire [7:0] mc32 = (xtime(sr02) ^ sr02) ^ sr12 ^ sr22 ^ xtime(sr32);

                // Column 3
                wire [7:0] mc03 = xtime(sr03) ^ (xtime(sr13) ^ sr13) ^ sr23 ^ sr32_3;
                wire [7:0] mc13 = sr03 ^ xtime(sr13) ^ (xtime(sr23) ^ sr23) ^ sr32_3;
                wire [7:0] mc23 = sr03 ^ sr13 ^ xtime(sr23) ^ (xtime(sr32_3) ^ sr32_3);
                wire [7:0] mc33 = (xtime(sr03) ^ sr03) ^ sr13 ^ sr23 ^ xtime(sr32_3);

                // AddRoundKey
                assign state[ri] = {mc00, mc10, mc20, mc30,
                                    mc01, mc11, mc21, mc31,
                                    mc02, mc12, mc22, mc32,
                                    mc03, mc13, mc23, mc33} ^ rk[ri];
            end else begin : last_round
                // Last round: no MixColumns, just SubBytes + ShiftRows + AddRoundKey
                assign state[ri] = {sr00, sr10, sr20, sr30,
                                    sr01, sr11, sr21, sr31,
                                    sr02, sr12, sr22, sr32,
                                    sr03, sr13, sr23, sr32_3} ^ rk[ri];
            end
        end
    endgenerate

    assign ciphertext = state[10];

endmodule
