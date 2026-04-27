for solid_mesh in 5
do
    for fluid_mesh in 128
    do
        for time_step in 10000
        do
            for mu_s in 0.1 0.2 0.5 1.0 2.0 10.0
            do
                echo "solid_mesh: $solid_mesh, fluid_mesh: $fluid_mesh, time_step: $time_step, mu_s: $mu_s_"
                echo "nohup_demo_DrivenDisk_2D_explicit_$solid_mesh_$fluid_mesh_$time_step_100_$mu_s.txt"
                # nohup ./test/explicit_ibm $solid_mesh $fluid_mesh $time_step 100 $mu_s >> "nohup_demo_DrivenDisk_2D_explicit_$solid_mesh_$fluid_mesh_$time_step_100_$mu_s.txt" &
                # 在这里执行命令，可以使用$variable引用当前迭代的元素
            done
        done
    done
done
# bash ./run.bash