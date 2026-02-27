// T7: Module instantiation - 2-bit adder from two full adders
module full_adder(
    input a,
    input b,
    input cin,
    output sum,
    output cout
);
    assign sum = a ^ b ^ cin;
    assign cout = (a & b) | (b & cin) | (a & cin);
endmodule

module adder2(
    input [1:0] a,
    input [1:0] b,
    output [2:0] sum
);
    wire c0;

    full_adder fa0(.a(a[0]), .b(b[0]), .cin(1'b0), .sum(sum[0]), .cout(c0));
    full_adder fa1(.a(a[1]), .b(b[1]), .cin(c0), .sum(sum[1]), .cout(sum[2]));
endmodule
