import YAML

function read_lines(callback, file)
    open(file) do io
        while !eof(io)
            line = readline(io)
            callback(line)
        end
    end
end

struct Op
    name::String
    meta::String
end

ops = Set{Op}()

read_lines("/home/medavies/bert_large_s384_infer_bs1.yaml") do line

    try
        yd = YAML.load(line)[1]
        push!(ops, Op(yd["opname"], ""))
    catch
        println("ERROR: something went wrong with")
        println(line)
    end
end

for op in ops
    println(op)
end
