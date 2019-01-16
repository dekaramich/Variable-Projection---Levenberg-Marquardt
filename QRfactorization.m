function [Q1, Q2, R1] = QRfactorization(phi, m, N);

    %QR factorization
    [Q,R] = qr(phi); %Q*R = phi

    %QR "thin" factorization, Q1*R1 = phi
    %Q1 ( m x N)
    %( N + 1 because of the bias )
    Q1 = Q(1:m,1:N+1);
    %Q1 ( m x (m-N) )
    Q2 = Q(1:m, N+2:m);

    %Qtest = [ Q1 Q2 ];

    %R1 (N x N)
    R1 = R(1:N+1,1:N+1);
    
end