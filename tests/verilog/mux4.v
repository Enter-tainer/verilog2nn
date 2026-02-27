// T6: 4-to-1 MUX
module mux4(
    input [1:0] sel,
    input d0,
    input d1,
    input d2,
    input d3,
    output y
);
    assign y = (sel == 2'd0) ? d0 :
               (sel == 2'd1) ? d1 :
               (sel == 2'd2) ? d2 : d3;
endmodule
