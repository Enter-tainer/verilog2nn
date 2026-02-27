// T3: Combination expression
module combo_expr(input a, input b, input c, output y);
    assign y = (a & b) | (~c);
endmodule
