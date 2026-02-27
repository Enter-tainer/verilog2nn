// T11: MD5 single round function (F function + one round step)
// MD5 F function: F(B,C,D) = (B & C) | (~B & D)
// One round: A' = D, B' = B + ((A + F(B,C,D) + M + K) <<< s), C' = B, D' = C
// Simplified: just the F function + addition + rotate for one step

module md5_round(
    input [31:0] a,
    input [31:0] b,
    input [31:0] c,
    input [31:0] d,
    input [31:0] m,   // message word
    input [31:0] k,   // round constant
    output [31:0] a_out,
    output [31:0] b_out,
    output [31:0] c_out,
    output [31:0] d_out
);
    // F function
    wire [31:0] f = (b & c) | (~b & d);

    // Sum
    wire [31:0] sum = a + f + m + k;

    // Rotate left by 7 (fixed shift for simplicity)
    wire [31:0] rotated = {sum[24:0], sum[31:25]};

    // Output
    assign a_out = d;
    assign b_out = b + rotated;
    assign c_out = b;
    assign d_out = c;
endmodule
