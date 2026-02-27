// T5: 8-bit comparator
module cmp8(
    input [7:0] a,
    input [7:0] b,
    output gt,
    output eq,
    output lt
);
    assign gt = (a > b);
    assign eq = (a == b);
    assign lt = (a < b);
endmodule
